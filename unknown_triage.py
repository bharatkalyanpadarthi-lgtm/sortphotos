#!/usr/bin/env python3
"""
Create a compact HTML/CSV triage report for unlabeled face clusters.

This does not move or relabel anything. It helps you see which unknown clusters
are worth naming by grouping samples, source folders, and face quality.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sort_photos  # noqa: E402

for _name in ("CacheState", "CachedFace", "FaceRecord", "LabelingState", "IdentityDB"):
    if hasattr(sort_photos, _name):
        setattr(sys.modules["__main__"], _name, getattr(sort_photos, _name))

DEFAULT_STATE = Path.home() / ".face_sort_cache" / "labeling_state.pkl"
DEFAULT_OUTPUT_DIR = (
    Path.home() / "Pictures" / "sorted_all_pictures" / "_source_review" / "unknown_triage"
)


def load_state(path: Path):
    if not path.exists():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


def parent_bucket(path: Path, input_dir: Path | None) -> str:
    try:
        if input_dir is not None:
            rel = path.relative_to(input_dir)
            return rel.parts[0] if len(rel.parts) > 1 else "."
    except ValueError:
        pass
    return path.parent.name


def crop_data_uri(crop_jpeg: bytes | None) -> str:
    if not crop_jpeg:
        return ""
    encoded = base64.b64encode(crop_jpeg).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def face_src(face) -> Path:
    if hasattr(face, "src"):
        return Path(face.src)
    return Path(getattr(face, "src_str", ""))


def build_clusters(state, min_faces: int, max_samples: int) -> list[dict]:
    input_dir = Path(state.input_dir).expanduser() if getattr(state, "input_dir", "") else None
    clusters: dict[int, list] = defaultdict(list)
    for face, cid in zip(state.faces, state.cluster_ids):
        if cid == -1:
            continue
        label = state.name_map.get(cid, "")
        if label.startswith("person_"):
            clusters[int(cid)].append(face)

    out = []
    for cid, faces in clusters.items():
        if len(faces) < min_faces:
            continue
        faces_sorted = sorted(
            faces,
            key=lambda f: (-float(getattr(f, "quality", 0.0)), -float(getattr(f, "sharpness", 0.0)), str(face_src(f))),
        )
        sources = Counter(parent_bucket(face_src(f), input_dir) for f in faces)
        out.append({
            "cluster_id": cid,
            "label": state.name_map.get(cid, f"person_{cid}"),
            "face_count": len(faces),
            "source_count": len({str(face_src(f)) for f in faces}),
            "top_sources": sources.most_common(5),
            "avg_quality": sum(float(getattr(f, "quality", 0.0)) for f in faces) / max(1, len(faces)),
            "avg_sharpness": sum(float(getattr(f, "sharpness", 0.0)) for f in faces) / max(1, len(faces)),
            "samples": faces_sorted[:max_samples],
        })
    out.sort(key=lambda c: (-c["face_count"], -c["avg_quality"], c["label"]))
    return out


def write_csv(path: Path, clusters: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cluster_id", "label", "face_count", "source_count",
            "avg_quality", "avg_sharpness", "top_sources",
        ])
        for c in clusters:
            writer.writerow([
                c["cluster_id"],
                c["label"],
                c["face_count"],
                c["source_count"],
                f"{c['avg_quality']:.4f}",
                f"{c['avg_sharpness']:.1f}",
                "; ".join(f"{name}:{count}" for name, count in c["top_sources"]),
            ])


def write_html(path: Path, clusters: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sections = []
    for c in clusters:
        samples = []
        for face in c["samples"]:
            uri = crop_data_uri(getattr(face, "crop_jpeg", b""))
            src = html.escape(str(face_src(face)))
            img = f'<img src="{uri}" alt="face crop">' if uri else '<div class="noimg">no crop</div>'
            samples.append(f"""
              <figure>
                {img}
                <figcaption>q={float(getattr(face, "quality", 0.0)):.2f}<br>{src}</figcaption>
              </figure>
            """)
        top_sources = ", ".join(f"{html.escape(name)} ({count})" for name, count in c["top_sources"])
        sections.append(f"""
          <section>
            <header>
              <h2>{html.escape(c["label"])} · cluster {c["cluster_id"]}</h2>
              <p>{c["face_count"]} faces · {c["source_count"]} source images · avg quality {c["avg_quality"]:.2f}</p>
              <p>Sources: {top_sources}</p>
            </header>
            <div class="grid">{"".join(samples)}</div>
          </section>
        """)
    body = "".join(sections) if sections else "<p class='empty'>No unlabeled clusters matched this threshold.</p>"
    path.write_text(f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Unknown Face Triage</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f7f7f4; color: #242420; }}
    .top {{ position: sticky; top: 0; background: rgba(247,247,244,.95); border-bottom: 1px solid #d9d7cf; padding: 14px 22px; }}
    h1 {{ margin: 0; font-size: 22px; }}
    main {{ padding: 0 22px 36px; }}
    section {{ border-top: 1px solid #d9d7cf; padding: 18px 0 24px; }}
    h2 {{ margin: 0; font-size: 18px; }}
    p {{ margin: 4px 0; color: #686861; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 12px; margin-top: 12px; }}
    figure {{ margin: 0; background: white; border: 1px solid #ddd9cf; border-radius: 8px; overflow: hidden; }}
    img, .noimg {{ width: 100%; height: 150px; object-fit: contain; background: #eeeae1; display: flex; align-items: center; justify-content: center; }}
    figcaption {{ padding: 7px; color: #666; font-size: 11px; overflow-wrap: anywhere; }}
    .empty {{ padding: 40px 0; }}
  </style>
</head>
<body>
  <div class="top"><h1>Unknown Face Triage</h1></div>
  <main>{body}</main>
</body>
</html>
""", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-faces", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=12)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    state = load_state(args.state.expanduser())
    output_dir = args.output_dir.expanduser().resolve()
    if state is None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "unknown_triage.csv").write_text(
            "cluster_id,label,face_count,source_count,avg_quality,avg_sharpness,top_sources\n",
            encoding="utf-8",
        )
        write_html(output_dir / "unknown_triage.html", [])
        if not args.quiet:
            print("No saved labeling state found; no unknown triage needed.")
        return 0

    clusters = build_clusters(state, max(1, int(args.min_faces)), max(1, int(args.max_samples)))
    csv_path = output_dir / "unknown_triage.csv"
    html_path = output_dir / "unknown_triage.html"
    write_csv(csv_path, clusters)
    write_html(html_path, clusters)
    print(f"Unknown clusters: {len(clusters)}")
    print(f"CSV:              {csv_path}")
    print(f"HTML:             {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
