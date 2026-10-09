"""Quick Review presentation; kept outside recognition safety fingerprints."""

from __future__ import annotations

import html
import json
from urllib.parse import urlencode

import identity_profiles

def render_html(
    clusters: list[UnknownCluster],
    decisions: dict,
    identity_names: list[str],
    summary: dict,
    *,
    interactive: bool,
    decision_for_item,
    candidate_summary,
    MAX_CLUSTER_ACTION_ITEMS: int,
) -> str:
    cards: list[str] = []
    pending_clusters = 0
    pending_items = 0
    known_people_json = json.dumps(
        [name.casefold() for name in identity_names], ensure_ascii=True
    ).replace("</", "<\\/")
    all_people = "".join(
        f'<option value="{html.escape(name, quote=True)}"></option>'
        for name in identity_names
    )
    for cluster in clusters:
        unresolved = [
            item for item in cluster.items
            if not decision_for_item(decisions, item).get("action")
            and item.path.is_file()
        ]
        resolved = not unresolved
        if unresolved:
            pending_clusters += 1
            pending_items += len(unresolved)
        best_name = cluster.candidates[0].name if cluster.candidates else ""
        cluster_disabled = " disabled" if not interactive or not unresolved else ""
        suggestions = "".join(
            "<button class='suggestion secondary-button' data-action='confirm' data-scope='cluster' "
            f"data-shortcut='{index}' title='Confirm all as {html.escape(candidate.name, quote=True)}' "
            f"data-person='{html.escape(candidate.name, quote=True)}'{cluster_disabled}>"
            f"<kbd>{index}</kbd> {html.escape(candidate.name)} "
            f"<span>{candidate.distance:.3f}</span></button>"
            for index, candidate in enumerate(cluster.candidates, 1)
        )
        matcher_agreement = bool(
            unresolved
            and best_name
            and all(
                item.candidates
                and item.candidates[0].name.casefold() == best_name.casefold()
                and item.secondary is not None
                and item.secondary.accepted
                and str(item.secondary.predicted or "").casefold() == best_name.casefold()
                for item in unresolved
            )
        )
        agreement_badge = (
            "<span class='agreement-badge'>Both matchers agree; held for manual review by the remaining safety rules</span>"
            if matcher_agreement else ""
        )
        item_cards: list[str] = []
        for item in cluster.items:
            decision = decision_for_item(decisions, item)
            decided_action = str(decision.get("action", ""))
            exists = item.path.is_file()
            item_best = item.candidates[0].name if item.candidates else best_name
            face_url = "/face?" + urlencode({"item": item.key})
            full_url = "/image?" + urlencode({"item": item.key})
            disabled = " disabled" if not interactive or decided_action or not exists else ""
            selectable_disabled = " disabled" if decided_action or not exists else ""
            status = {
                "confirmed": f"Confirmed as {decision.get('person', '')}",
                "keep_unknown": "Kept as unknown",
                "ignored": "Ignored",
                "moved_to_junk": "Moved to recoverable junk",
            }.get(decided_action, "Pending" if exists else "File moved")
            item_margin = identity_profiles.candidate_margin(item.candidates)
            rejected_people = {
                str(value) for value in decision.get("rejected_people", []) if str(value)
            }
            secondary_text = ""
            if item.secondary is not None:
                result = "agrees" if item.secondary.accepted else "does not agree"
                secondary_text = (
                    f" · secondary {result}: {item.secondary.predicted or 'none'} "
                    f"d={item.secondary.distance:.3f} m={item.secondary.margin:.3f}"
                )
            negative_actions = "".join(
                "<button class='negative secondary-button' data-action='reject_candidate' data-scope='item' "
                f"data-person='{html.escape(candidate.name, quote=True)}'{disabled}>"
                f"Not {html.escape(candidate.name)}</button>"
                for candidate in item.candidates
                if candidate.name not in rejected_people
            )
            candidate_chips = "".join(
                f"<span class='candidate-chip'>{html.escape(candidate.name)} "
                f"<b>{candidate.distance:.3f}</b></span>"
                for candidate in item.candidates
            )
            item_search = " ".join((
                item.path.name,
                str(item.path),
                " ".join(candidate.name for candidate in item.candidates),
                status,
            )).casefold()
            item_status = "reviewed" if decided_action or not exists else "pending"
            reasons = summary.get("review_reasons", {}).get(item.key, [])
            if item_best.casefold() in {name.casefold() for name in rejected_people}:
                reasons = [{"code": "user_rejected_candidate", "message": f"You rejected {item_best} for this image."}]
            reason_html = "".join(
                f"<p class='review-reason' data-reason='{html.escape(reason['code'], quote=True)}'>"
                f"{html.escape(reason['message'])}</p>"
                for reason in reasons
            ) if not decided_action else ""
            item_cards.append(
                f"<article class='item {item_status}' data-item='{item.key}' "
                f"data-status='{item_status}' data-best='{html.escape(item_best, quote=True)}' "
                f"data-search='{html.escape(item_search, quote=True)}'>"
                f"<div class='image-area'><button class='preview-button' type='button' "
                f"data-full='{html.escape(full_url, quote=True)}' "
                f"data-name='{html.escape(item.path.name, quote=True)}'>"
                f"<img loading='lazy' decoding='async' src='{html.escape(face_url, quote=True)}' alt='Detected face'></button>"
                f"<label class='select-control' title='Select image'><input class='row-select' "
                f"type='checkbox' value='{item.key}'{selectable_disabled}><span></span></label>"
                f"<span class='status-badge'>{html.escape(status)}</span></div>"
                f"<div class='item-body'><strong class='filename' title='{html.escape(str(item.path), quote=True)}'>{html.escape(item.path.name)}</strong>"
                f"<div class='candidate-chips'>{candidate_chips}</div>"
                f"{reason_html}"
                f"<p class='evidence'>margin={item_margin:.3f} &middot; quality={item.face.quality:.3f} &middot; pose={html.escape(item.face.pose_label)}{html.escape(secondary_text)}</p>"
                f"</div><details class='item-review'><summary>Review actions</summary><div class='review-body'>"
                f"<label>Confirmed person<input class='person' list='people' value='{html.escape(item_best, quote=True)}' aria-label='Confirmed person'{disabled}></label>"
                f"<div class='item-actions'><button class='primary-button' data-action='confirm' data-scope='item'{disabled}>Confirm Cluster</button>"
                f"<button class='secondary-button' data-action='keep_unknown' data-scope='item'{disabled}>Keep Unknown</button>"
                f"<button class='secondary-button' data-action='ignore' data-scope='item'{disabled}>Ignore</button></div>"
                f"<div class='negative-actions'>{negative_actions}</div></div></details></article>"
            )
        cluster_search = " ".join((
            " ".join(candidate.name for candidate in cluster.candidates),
            " ".join(item.path.name for item in cluster.items),
        )).casefold()
        cluster_kind = "group" if len(cluster.items) > 1 else "single"
        cards.append(
            f"<section class='cluster {'resolved-cluster' if resolved else ''}' data-cluster='{cluster.key}' "
            f"data-kind='{cluster_kind}' data-search='{html.escape(cluster_search, quote=True)}' "
            f"data-pending='{'0' if resolved else '1'}' "
            f"data-items='{html.escape(','.join(item.key for item in unresolved), quote=True)}'>"
            f"<div class='cluster-head'><div><h2>{len(cluster.items)} similar face(s)</h2>"
            f"<p>{html.escape(candidate_summary(cluster))}</p>"
            f"<p>Cohesion {cluster.cohesion:.3f}; every image in this visual cluster is shown.</p>{agreement_badge}</div>"
            f"<div class='cluster-actions'><button class='secondary-button select-cluster' type='button'{cluster_disabled}>Select cluster</button>"
            f"<button class='secondary-button' data-action='keep_unknown' data-scope='cluster'{cluster_disabled}><kbd>U</kbd> Confirm all as Unknown</button>"
            f"<button class='danger-button' data-action='move_to_junk' data-scope='cluster'{cluster_disabled}><kbd>J</kbd> Move all to Junk</button>"
            f"{suggestions}<div class='custom-person'><input class='cluster-person' list='people' placeholder='Existing or new person' aria-label='Person for this cluster'{cluster_disabled}>"
            f"<button class='primary-button confirm-custom' type='button'{cluster_disabled}>Confirm Cluster</button></div></div></div>"
            f"<div class='items'>{''.join(item_cards)}</div></section>"
        )
    mode_note = (
        "Actions are live. Confirmed files are copied and verified before the unknown source is archived recoverably."
        if interactive
        else "Static preview; launch from Face Terminal to use review actions."
    )
    queue_files = summary.get("queue_files", 0)
    batch_files = summary.get("batch_files", 0)
    sqlite_hits = summary.get("sqlite_hits", 0)
    detected = summary.get("detected", 0)
    progress = summary.get("progress", {})
    reviewed = int(progress.get("reviewed", 0))
    total = int(progress.get("total", queue_files))
    confirmed = int(progress.get("confirmed", 0))
    unknown = int(progress.get("unknown", 0))
    junk = int(progress.get("junk", 0))
    deferred = int(progress.get("deferred", 0))
    auto_review = summary.get("auto_review", {})
    auto_sweep = summary.get("auto_sweep", {})
    auto_confirmed = int(auto_review.get("confirmed", 0)) if isinstance(auto_review, dict) else 0
    auto_eligible = int(auto_review.get("eligible", 0)) if isinstance(auto_review, dict) else 0
    auto_preview = bool(auto_review.get("preview")) if isinstance(auto_review, dict) else False
    auto_enabled = bool(auto_review.get("enabled")) if isinstance(auto_review, dict) else False
    auto_requested = bool(auto_review.get("requested")) if isinstance(auto_review, dict) else False
    auto_gate_message = str(auto_review.get("gate_message", "")) if isinstance(auto_review, dict) else ""
    sweep_scanned = int(auto_sweep.get("scanned", 0)) if isinstance(auto_sweep, dict) else 0
    sweep_confirmed = int(auto_sweep.get("confirmed", 0)) if isinstance(auto_sweep, dict) else 0
    sweep_cached = bool(auto_sweep.get("cached")) if isinstance(auto_sweep, dict) else False
    if auto_preview:
        auto_status = f"Safe auto-match preview: {auto_eligible} eligible"
    elif sweep_scanned:
        auto_status = (
            f"Safe sweep: {sweep_confirmed} filed from {sweep_scanned} scanned"
            + (" (already current)" if sweep_cached else "")
        )
    elif auto_enabled:
        auto_status = f"Safe auto-match: {auto_confirmed} filed this batch"
    elif auto_requested:
        auto_status = auto_gate_message or "Safe auto-match unavailable"
    else:
        auto_status = "Safe auto-match off"
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width"><title>Unknown Identity Review</title>
<style>
:root{{color-scheme:dark;--bg:#0b0c0e;--panel:#17191d;--panel2:#22252b;--line:#343840;--text:#f5f5f7;--muted:#a8adb5;--blue:#0a84ff;--green:#30d158;--amber:#ffd60a;--red:#ff6961}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px -apple-system,BlinkMacSystemFont,"SF Pro Text",sans-serif;letter-spacing:0}}
button,input,select{{font:inherit}}button{{cursor:pointer}}button:disabled,input:disabled{{opacity:.35;cursor:not-allowed}}
.review-reason{{color:var(--amber);font-size:13px;line-height:1.4;overflow-wrap:anywhere;margin:6px 0}}
kbd{{display:inline-grid;place-items:center;min-width:20px;height:20px;padding:0 5px;border:1px solid #5b606a;border-radius:4px;background:#2c3036;color:#fff;font:600 11px -apple-system,BlinkMacSystemFont,"SF Mono",monospace}}
header{{position:sticky;top:0;z-index:20;padding:15px 20px 12px;background:rgba(11,12,14,.95);border-bottom:1px solid var(--line);backdrop-filter:blur(18px)}}
.inner{{max-width:1800px;margin:auto}}h1{{font-size:25px;margin:0}}h2{{font-size:17px;margin:0 0 4px}}p{{margin:4px 0;color:var(--muted)}}
.metrics,.toolbar,.batch-row,.batch-actions,.cluster-head,.cluster-actions{{display:flex;align-items:center}}
.metrics{{gap:7px;flex-wrap:wrap;margin-top:10px}}.metric{{padding:5px 8px;border-radius:5px;background:var(--panel2);color:var(--muted)}}.metric strong{{color:var(--text)}}
.toolbar{{gap:8px;margin-top:10px;flex-wrap:wrap}}.toolbar input,.toolbar select,.batch-panel input{{min-height:36px;padding:7px 9px;border:1px solid var(--line);border-radius:6px;color:var(--text);background:var(--panel2)}}
.quick-nav{{display:flex;align-items:center;gap:8px;margin-top:10px;padding-top:10px;border-top:1px solid var(--line)}}.quick-nav .cluster-position{{min-width:130px;text-align:center;color:var(--muted)}}.quick-nav .finish-button{{margin-left:auto}}.shortcut-help{{color:var(--muted);font-size:12px}}
.search{{min-width:260px;flex:1}}.toolbar select{{width:auto}}input:focus,select:focus{{outline:2px solid var(--blue);outline-offset:-1px}}
.primary-button,.secondary-button,.danger-button{{display:inline-flex;align-items:center;justify-content:center;gap:6px;min-height:34px;padding:6px 9px;border:1px solid var(--line);border-radius:6px;background:var(--panel2);color:var(--text);text-decoration:none;white-space:nowrap}}
.primary-button{{border-color:var(--blue);background:var(--blue);color:white}}button[data-action=keep_unknown]{{color:#ffe680}}button.negative{{color:#ff9b96;font-size:11px;min-height:28px}}
.danger-button{{border-color:#873f3b;background:#3a1c1a;color:#ffb2ad}}
.density-switch{{display:flex;border:1px solid var(--line);border-radius:6px;overflow:hidden}}.density-switch button{{min-height:34px;padding:6px 9px;border:0;border-right:1px solid var(--line);background:var(--panel);color:var(--muted)}}.density-switch button:last-child{{border-right:0}}.density-switch button.active{{background:#3a3d45;color:white}}
.batch-panel{{display:none;margin-top:10px;padding:10px;border:1px solid #285987;border-radius:7px;background:#14263a}}.batch-panel.active{{display:block}}.batch-row{{justify-content:space-between;gap:12px}}.batch-actions{{gap:6px;flex-wrap:wrap}}.batch-person{{min-width:230px}}.batch-suggestion{{color:#b9d9ff}}
main{{max-width:1800px;margin:auto;padding:0 20px 42px}}.cluster{{padding:18px 0;border-bottom:1px solid var(--line)}}.cluster.hidden{{display:none}}body.quick-review .cluster{{display:none}}body.quick-review .cluster.quick-active{{display:block}}
.cluster-head{{justify-content:space-between;gap:18px;align-items:flex-start;margin-bottom:11px}}.cluster-actions{{gap:6px;flex-wrap:wrap;justify-content:flex-end}}.suggestion span{{color:#b5bac2}}
.custom-person{{display:flex;gap:6px}}.cluster-person{{min-width:220px;min-height:34px;padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:#0f1114;color:var(--text)}}.agreement-badge{{display:inline-block;margin-top:7px;padding:4px 7px;border-radius:4px;background:#173d25;color:#b5f4c3;font-size:12px;font-weight:700}}
.items{{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:9px}}body.comfortable .items{{grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:13px}}
.item{{min-width:0;overflow:hidden;border:1px solid var(--line);border-radius:7px;background:var(--panel);transition:border-color .12s,box-shadow .12s}}.item.hidden{{display:none}}.item.selected{{border-color:var(--blue);box-shadow:0 0 0 2px rgba(10,132,255,.3)}}.item.reviewed{{opacity:.52}}.item.processing{{border-color:#725f16;opacity:.68}}.item.processing .status-badge{{background:#725f16;color:#fff0a8}}
.image-area{{position:relative;aspect-ratio:1;background:#030304}}.preview-button{{width:100%;height:100%;padding:0;border:0;background:#030304}}.preview-button img{{display:block;width:100%;height:100%;object-fit:contain}}
.select-control{{position:absolute;top:7px;left:7px;width:30px;height:30px;padding:6px;border-radius:50%;background:rgba(0,0,0,.72)}}.select-control input{{position:absolute;opacity:0;pointer-events:none}}.select-control span{{display:block;width:18px;height:18px;border:2px solid white;border-radius:50%}}.select-control input:checked+span{{border-color:var(--blue);background:var(--blue)}}.select-control input:checked+span:after{{content:"";display:block;width:8px;height:4px;margin:4px 0 0 3px;border-left:2px solid white;border-bottom:2px solid white;transform:rotate(-45deg)}}
.status-badge{{position:absolute;top:7px;right:7px;max-width:70%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding:4px 6px;border-radius:4px;background:rgba(82,63,0,.9);color:#ffe680;font-size:11px;font-weight:700}}.reviewed .status-badge{{background:rgba(31,79,43,.9);color:#aff4bd}}
.item-body{{padding:9px}}.filename{{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}.candidate-chips{{display:flex;gap:4px;flex-wrap:wrap;margin-top:7px}}.candidate-chip{{padding:3px 5px;border-radius:4px;background:#16354f;color:#d0e9ff;font-size:11px}}.candidate-chip b{{color:white}}.evidence{{font-size:11px;line-height:1.35;min-height:30px;margin-top:7px}}
.item-review{{border-top:1px solid var(--line)}}.item-review summary{{padding:8px 9px;color:#b9d9ff;cursor:pointer}}.review-body{{padding:0 9px 9px}}.review-body label{{color:var(--muted)}}input.person{{display:block;width:100%;height:35px;margin:5px 0 8px;padding:0 8px;border:1px solid var(--line);border-radius:5px;background:#0f1114;color:var(--text)}}.item-actions{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:5px}}.negative-actions{{display:flex;gap:5px;flex-wrap:wrap;margin-top:7px}}
#toast{{display:none;position:fixed;right:18px;bottom:18px;max-width:460px;padding:12px 14px;border:1px solid var(--line);border-radius:7px;background:#22262b;z-index:40}}dialog{{width:min(92vw,1200px);height:min(92vh,920px);padding:0;border:1px solid #555;border-radius:8px;background:#050506;color:white}}dialog::backdrop{{background:rgba(0,0,0,.84)}}.viewer-bar{{height:48px;display:flex;align-items:center;justify-content:space-between;gap:12px;padding:0 12px;border-bottom:1px solid var(--line)}}.viewer-name{{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}.viewer-image{{display:block;width:100%;height:calc(100% - 48px);object-fit:contain;background:#000}}.confirm-dialog{{width:min(92vw,520px);height:auto;padding:18px}}.confirm-dialog form{{margin:0}}.confirm-dialog p{{font-size:16px;line-height:1.45}}.confirm-actions{{display:flex;justify-content:flex-end;gap:8px;margin-top:18px}}
@media(max-width:760px){{header,main{{padding-left:11px;padding-right:11px}}.cluster-head,.batch-row{{display:block}}.cluster-actions,.batch-actions{{justify-content:flex-start;margin-top:9px}}.quick-nav{{flex-wrap:wrap}}.quick-nav .finish-button{{margin-left:0}}.custom-person{{width:100%}}.cluster-person{{min-width:0;flex:1}}.items{{grid-template-columns:repeat(auto-fill,minmax(148px,1fr));gap:7px}}body.comfortable .items{{grid-template-columns:repeat(auto-fill,minmax(220px,1fr))}}.item-actions{{grid-template-columns:1fr}}}}
</style></head><body class="quick-review"><header><div class="inner"><h1>Unknown Identity Quick Review</h1><p>{html.escape(mode_note)}</p>
<div class="metrics"><span class="metric">Reviewed <strong id="reviewedCount">{reviewed}</strong> of <strong id="totalCount">{total}</strong></span><span class="metric">Confirmed <strong id="confirmedCount">{confirmed}</strong></span><span class="metric">Unknown <strong id="unknownCount">{unknown}</strong></span><span class="metric">Junk <strong id="junkCount">{junk}</strong></span><span class="metric">Deferred <strong id="deferredCount">{deferred}</strong></span><span class="metric"><strong>{pending_clusters}</strong> clusters in this batch</span><span class="metric">Action queue <strong id="queueCount">0</strong></span><span class="metric">Batch <strong>{summary.get('batch_number', 1)}</strong> ({batch_files} files)</span><span class="metric">Cache <strong>{sqlite_hits}</strong> reused / <strong>{detected}</strong> analyzed</span><span class="metric">{html.escape(auto_status)}</span><span id="headerMessage" class="batch-suggestion"></span></div>
<div class="toolbar"><input id="search" class="search" type="search" placeholder="Search filename or suggested person"><select id="statusFilter"><option value="pending">Pending only</option><option value="all">All status</option><option value="reviewed">Reviewed only</option></select><select id="clusterFilter"><option value="all">All clusters</option><option value="group">Similar groups</option><option value="single">Single images</option></select><button id="selectVisible" class="secondary-button" type="button">Select visible</button><button id="clearSelection" class="secondary-button" type="button">Clear</button><div class="density-switch"><button class="active" type="button" data-density="compact">Compact</button><button type="button" data-density="comfortable">Large</button></div></div>
<div class="quick-nav"><button id="previousCluster" class="secondary-button" type="button" title="Previous cluster (Left Arrow)">Previous</button><strong id="clusterPosition" class="cluster-position">Cluster 0 of 0</strong><button id="nextCluster" class="secondary-button" type="button" title="Next cluster (Right Arrow)">Next</button><button id="skipCluster" class="secondary-button" type="button" title="Skip this cluster for this session (N)"><kbd>N</kbd> Skip</button><button id="loadNextBatch" class="secondary-button" type="button">Load Next Batch</button><span class="shortcut-help"><kbd>1</kbd> <kbd>2</kbd> <kbd>3</kbd> suggestions &nbsp; <kbd>U</kbd> unknown &nbsp; <kbd>J</kbd> junk</span><button id="finishReview" class="primary-button finish-button" type="button">Finish Review</button></div>
<section id="batchPanel" class="batch-panel"><div class="batch-row"><div><strong><span id="selectedCount">0</span> images selected</strong><span id="batchSuggestion" class="batch-suggestion"></span></div><div class="batch-actions"><input id="batchPerson" class="batch-person" list="people" placeholder="Choose person to confirm"><button id="useSuggestion" class="secondary-button" type="button" hidden></button><button id="confirmSelected" class="primary-button" type="button">Confirm selected</button><button id="keepSelected" class="secondary-button" type="button">Keep Unknown</button><button id="ignoreSelected" class="secondary-button" type="button">Defer</button></div></div></section>
</div></header><main>{''.join(cards) or '<p id="emptyBatch">No reviewable single-face unknowns in this batch.</p>'}</main>
<datalist id="people">{all_people}</datalist><div id="toast"></div><dialog id="viewer"><div class="viewer-bar"><strong id="viewerName" class="viewer-name"></strong><button id="closeViewer" class="secondary-button" type="button">Close</button></div><img id="viewerImage" class="viewer-image" alt="Full unknown image"></dialog><dialog id="confirmDialog" class="confirm-dialog"><form method="dialog"><h2>Finish Unknown Review?</h2><p id="confirmMessage"></p><div class="confirm-actions"><button class="secondary-button" value="cancel">Cancel</button><button class="primary-button" value="confirm">Finish Review</button></div></form></dialog>
<script>
const interactive={str(interactive).lower()};
const toast=document.getElementById('toast');
const queueCount=document.getElementById('queueCount');
const search=document.getElementById('search');
const statusFilter=document.getElementById('statusFilter');
const clusterFilter=document.getElementById('clusterFilter');
const batchPanel=document.getElementById('batchPanel');
const batchPerson=document.getElementById('batchPerson');
const useSuggestion=document.getElementById('useSuggestion');
const activeJobIds=new Set(JSON.parse(sessionStorage.getItem('unknownJobs')||'[]'));
const temporarilySkipped=new Set();
let activeIndex=0,lastChecked=null,polling=false,queueRefreshPending=false,loadingBatch=false;
let lifecycleReady=false,finishing=false,reviewFinished=false,queueComplete=false;
let autoLoadPaused=false,waitingForActions=false,batchTimer=null,finishTimer=null,batchGeneration=0;
const allClusters=()=>[...document.querySelectorAll('.cluster')];
const cards=()=>[...document.querySelectorAll('.item')];
const checks=()=>[...document.querySelectorAll('.row-select')];
const selected=()=>checks().filter(input=>input.checked);
const availableClusters=()=>allClusters().filter(cluster=>
  cluster.dataset.pending==='1'&&!cluster.classList.contains('hidden')&&!temporarilySkipped.has(cluster.dataset.cluster)
);
const unresolvedClusters=()=>allClusters().filter(cluster=>
  cluster.dataset.pending==='1'&&!temporarilySkipped.has(cluster.dataset.cluster)
);
function updateLifecycleControls(){{
  document.getElementById('loadNextBatch').disabled=!lifecycleReady||finishing||reviewFinished||loadingBatch;
  document.getElementById('finishReview').disabled=!lifecycleReady||finishing||reviewFinished||loadingBatch;
  if(finishing||reviewFinished)document.querySelectorAll('.cluster button,.cluster input,#batchPanel button,#batchPanel input,#skipCluster').forEach(control=>control.disabled=true);
}}
function pauseBatchLoading(){{
  clearTimeout(batchTimer);batchTimer=null;batchGeneration++;loadingBatch=false;
}}
function finishState(result){{
  const button=document.getElementById('finishReview');
  if(result.status==='completed'){{
    reviewFinished=true;finishing=false;queueComplete=true;pauseBatchLoading();clearTimeout(finishTimer);
    button.textContent='Review Finished';document.getElementById('clusterPosition').textContent='Review Finished';
    updateLifecycleControls();pollJobs(true);return true;
  }}
  if(result.status==='queued'||result.status==='running'){{
    finishing=true;pauseBatchLoading();button.textContent=result.step||'Finishing...';updateLifecycleControls();pollJobs(true);return false;
  }}
  return false;
}}

function notify(message,error=false){{
  toast.textContent=message;toast.style.display='block';toast.style.borderColor=error?'var(--red)':'var(--line)';
  setTimeout(()=>toast.style.display='none',6000);
}}
function askConfirmation(message){{
  const dialog=document.getElementById('confirmDialog');
  document.getElementById('confirmMessage').textContent=message;dialog.returnValue='cancel';
  return new Promise(resolve=>{{dialog.addEventListener('close',()=>resolve(dialog.returnValue==='confirm'),{{once:true}});dialog.showModal();}});
}}
function persistJobs(){{sessionStorage.setItem('unknownJobs',JSON.stringify([...activeJobIds]));}}
function activeCluster(){{return document.querySelector('.cluster.quick-active');}}
function showCluster(index=activeIndex){{
  const available=availableClusters();allClusters().forEach(cluster=>cluster.classList.remove('quick-active'));
  if(!available.length){{if(!finishing&&!reviewFinished)document.getElementById('clusterPosition').textContent=queueComplete?'Queue complete':unresolvedClusters().length?'No clusters match these filters':'No pending cluster';maybeLoadNextBatch();return;}}
  activeIndex=((index%available.length)+available.length)%available.length;
  available[activeIndex].classList.add('quick-active');
  document.getElementById('clusterPosition').textContent=`Cluster ${{activeIndex+1}} of ${{available.length}}`;
  window.scrollTo({{top:0,behavior:'smooth'}});
}}
function moveCluster(delta){{showCluster(activeIndex+delta);}}
function updateSelection(){{
  cards().forEach(card=>{{const checkbox=card.querySelector('.row-select');card.classList.toggle('selected',Boolean(checkbox&&checkbox.checked));}});
  const chosen=selected();document.getElementById('selectedCount').textContent=chosen.length;batchPanel.classList.toggle('active',chosen.length>0);
  const best=[...new Set(chosen.map(input=>input.closest('.item').dataset.best).filter(Boolean))];
  const suggestion=best.length===1?best[0]:'';
  document.getElementById('batchSuggestion').textContent=suggestion?` · common suggestion: ${{suggestion}}`:best.length>1?' · mixed suggestions':'';
  useSuggestion.hidden=!suggestion;useSuggestion.textContent=suggestion?`Use ${{suggestion}}`:'';useSuggestion.dataset.person=suggestion;
}}
function applyFilters(){{
  const query=search.value.trim().toLocaleLowerCase(),status=statusFilter.value,kind=clusterFilter.value;
  allClusters().forEach(cluster=>{{
    const statusAllowed=status==='all'||(status==='pending'&&cluster.dataset.pending==='1')||(status==='reviewed'&&cluster.dataset.pending==='0');
    const allowed=(kind==='all'||cluster.dataset.kind===kind)&&statusAllowed&&(!query||cluster.dataset.search.includes(query));
    cluster.classList.toggle('hidden',!allowed);
  }});
  localStorage.setItem('unknownSearch',search.value);localStorage.setItem('unknownStatus',status);localStorage.setItem('unknownCluster',kind);
  activeIndex=0;showCluster(0);
}}
function markJobCards(job){{
  const label=job.status==='running'?'Working':job.position>1?`Queued #${{job.position}}`:'Queued';
  job.item_keys.forEach(key=>{{const card=document.querySelector(`.item[data-item="${{key}}"]`);if(!card)return;card.classList.add('processing');const checkbox=card.querySelector('.row-select');if(checkbox){{checkbox.checked=false;checkbox.disabled=true;}}const badge=card.querySelector('.status-badge');if(badge)badge.textContent=label;}});
  const first=document.querySelector(`.item[data-item="${{job.item_keys[0]}}"]`);const cluster=first?.closest('.cluster');
  if(cluster){{cluster.dataset.pending='0';cluster.querySelectorAll('button,input').forEach(control=>control.disabled=true);}}
  updateSelection();
}}
function optimisticProgress(action,count){{
  const increment=id=>{{const node=document.getElementById(id);node.textContent=String(Number(node.textContent||0)+count);}};
  increment('reviewedCount');
  if(action==='confirm')increment('confirmedCount');else if(action==='keep_unknown')increment('unknownCount');else if(action==='move_to_junk')increment('junkCount');else increment('deferredCount');
}}
async function refreshProgress(){{
  try{{const response=await fetch('/progress');const result=await response.json();if(!response.ok)return;for(const [key,id] of Object.entries({{reviewed:'reviewedCount',total:'totalCount',confirmed:'confirmedCount',unknown:'unknownCount',junk:'junkCount',deferred:'deferredCount'}})){{document.getElementById(id).textContent=result.progress[key]??0;}}}}catch(_error){{}}
}}
function applyResolvedItems(keys){{
  keys.forEach(key=>{{const card=document.querySelector(`.item[data-item="${{key}}"]`);if(!card)return;card.classList.add('processing');card.querySelectorAll('button,input').forEach(control=>{{control.disabled=true;if(control.type==='checkbox')control.checked=false;}});const badge=card.querySelector('.status-badge');if(badge)badge.textContent='Decision saved';}});
  allClusters().forEach(cluster=>{{if(!cluster.querySelector('.row-select:not(:disabled)'))cluster.dataset.pending='0';}});
  updateSelection();
}}
async function pollJobs(statusOnly=false){{
  if(polling){{if(statusOnly)queueRefreshPending=true;return;}}
  if((finishing||reviewFinished)&&!statusOnly)return;polling=true;let repeat=false,failed=false;
  try{{
    const ids=[...activeJobIds];const jobsURL=ids.length?'/jobs?'+new URLSearchParams({{ids:ids.join(',')}}):'/jobs';const response=await fetch(jobsURL);const result=await response.json();
    if(!response.ok)throw new Error(result.error||'Could not read action queue');
    queueCount.textContent=(result.summary.queued||0)+(result.summary.running||0);
    applyResolvedItems(result.resolved_item_keys||[]);
    const returnedJobIds=new Set(result.jobs.map(job=>job.id));
    ids.filter(id=>!returnedJobIds.has(id)).forEach(id=>activeJobIds.delete(id));
    result.jobs.forEach(job=>{{if(job.status==='queued'||job.status==='running'){{activeJobIds.add(job.id);markJobCards(job);}}else{{activeJobIds.delete(job.id);if(job.status==='failed'){{failed=true;notify(job.error||'Review action failed',true);}}else if(job.message)notify(job.message);}}}});
    persistJobs();repeat=activeJobIds.size>0||(result.summary.queued||0)+(result.summary.running||0)>0;
    // Finish still drains the queue, but must never resume batches or reload the page.
    if(statusOnly||finishing||reviewFinished)return;
    if(failed){{setTimeout(()=>location.reload(),900);return;}}
    if(!repeat){{waitingForActions=false;await refreshProgress();maybeLoadNextBatch();}}
  }}catch(error){{notify(error.message,true);repeat=true;}}
  finally{{
    polling=false;
    if(queueRefreshPending){{queueRefreshPending=false;pollJobs(true);}}
    else if(repeat&&!statusOnly&&!finishing&&!reviewFinished)setTimeout(pollJobs,700);
  }}
}}
async function decide(action,keys,person=''){{
  if(!interactive){{notify('Launch with face unknown-review to use actions.',true);return;}}
  if(!lifecycleReady||finishing||reviewFinished||loadingBatch)return;
  const values=Array.isArray(keys)?keys:keys.split(',').filter(Boolean);
  if(!values.length){{notify('This cluster has no pending images.',true);return;}}
  if(values.length>{MAX_CLUSTER_ACTION_ITEMS}){{notify('Select no more than {MAX_CLUSTER_ACTION_ITEMS} images.',true);return;}}
  if(action==='confirm'&&!person.trim()){{notify('Enter or choose a person.',true);return;}}
  try{{
    const response=await fetch('/decide',{{method:'POST',headers:{{'Content-Type':'application/x-www-form-urlencoded'}},body:new URLSearchParams({{action,person,item_keys:values.join(',')}})}});
    const result=await response.json();if(!response.ok)throw new Error(result.error||'Action failed');
    const jobs=result.jobs||[result.job];jobs.forEach(job=>{{activeJobIds.add(job.id);markJobCards(job);}});persistJobs();optimisticProgress(action,values.length);
    const firstJob=jobs[0];notify(jobs.length>1?`Queued safely in ${{jobs.length}} verified chunks.`:firstJob.position>1?`Queued safely at position ${{firstJob.position}}.`:'Action started.');showCluster(activeIndex);pollJobs();
  }}catch(error){{notify(error.message,true);}}
}}
async function skipCurrent(){{
  if(!interactive||!lifecycleReady||finishing||reviewFinished||loadingBatch)return;
  const cluster=activeCluster();if(!cluster)return;const keys=cluster.dataset.items.split(',').filter(Boolean);
  try{{const response=await fetch('/skip',{{method:'POST',headers:{{'Content-Type':'application/x-www-form-urlencoded'}},body:new URLSearchParams({{item_keys:keys.join(',')}})}});const result=await response.json();if(!response.ok)throw new Error(result.error||'Could not skip cluster');temporarilySkipped.add(cluster.dataset.cluster);notify('Skipped for this session.');showCluster(activeIndex);}}catch(error){{notify(error.message,true);}}
}}
async function maybeLoadNextBatch(manual=false){{
  if(!interactive||!lifecycleReady||loadingBatch||finishing||reviewFinished||waitingForActions)return;
  if(!manual&&(queueComplete||autoLoadPaused))return;
  if(unresolvedClusters().length){{if(manual)notify('Finish or skip the current pending clusters first.');return;}}
  autoLoadPaused=false;queueComplete=false;loadingBatch=true;const generation=++batchGeneration;
  updateLifecycleControls();document.getElementById('clusterPosition').textContent='Loading next batch...';
  try{{
    const response=await fetch('/next-batch',{{method:'POST'}});const result=await response.json();
    if(generation!==batchGeneration||finishing||reviewFinished)return;
    if(result.reason==='review_finished'){{finishState({{status:'completed'}});refreshProgress();return;}}
    if(response.status===409){{
      loadingBatch=false;updateLifecycleControls();document.getElementById('clusterPosition').textContent=result.error||'Waiting for current operation...';
      if(result.reason==='finishing'){{finishState({{status:'running'}});pollFinish();}}
      else if(result.reason==='actions_pending'){{waitingForActions=true;pollJobs();}}
      else{{autoLoadPaused=true;notify(result.error||'Loading paused. Please retry when ready.',true);}}
      return;
    }}
    if(!response.ok)throw new Error(result.error||'Could not load next batch');
    pollBatchStatus(generation);
  }}catch(error){{
    if(generation!==batchGeneration||finishing||reviewFinished)return;
    loadingBatch=false;autoLoadPaused=true;updateLifecycleControls();notify(error.message,true);document.getElementById('clusterPosition').textContent='Load failed';
  }}
}}
async function pollBatchStatus(generation=batchGeneration){{
  if(generation!==batchGeneration||finishing||reviewFinished)return;
  try{{
    const response=await fetch('/batch-status');const result=await response.json();
    if(generation!==batchGeneration||finishing||reviewFinished)return;
    if(!response.ok)throw new Error(result.error||'Could not read batch progress');
    document.getElementById('clusterPosition').textContent=result.step||'Loading next batch...';
    if(result.status==='completed'){{
      loadingBatch=false;queueComplete=Boolean(result.complete);updateLifecycleControls();
      if(queueComplete){{document.getElementById('clusterPosition').textContent='Queue complete';notify('All pending unknown files are handled. Finish Review when ready.');}}
      else{{location.reload();}}return;
    }}
    if(result.status==='failed'){{loadingBatch=false;autoLoadPaused=true;updateLifecycleControls();notify(result.message||'Could not load next batch',true);return;}}
    batchTimer=setTimeout(()=>pollBatchStatus(generation),500);
  }}catch(error){{
    if(generation!==batchGeneration||finishing||reviewFinished)return;
    document.getElementById('clusterPosition').textContent='Reconnecting to batch loader...';batchTimer=setTimeout(()=>pollBatchStatus(generation),1500);
  }}
}}
async function pollFinish(){{
  if(reviewFinished)return;clearTimeout(finishTimer);
  try{{
    const response=await fetch('/finish-status');const result=await response.json();if(reviewFinished)return;
    if(!response.ok)throw new Error(result.error||'Could not read finish status');
    if(finishState(result)){{notify(`${{result.message||'Review saved.'}} Report: ${{result.report||''}}`);await refreshProgress();return;}}
    if(result.status==='failed'||result.status==='idle'){{
      finishing=false;autoLoadPaused=true;document.getElementById('finishReview').textContent='Retry Finish Review';updateLifecycleControls();
      notify(result.message||'Finish was not started. Retry when ready.',true);return;
    }}
    finishTimer=setTimeout(pollFinish,1000);
  }}catch(error){{if(!reviewFinished){{notify(error.message,true);finishTimer=setTimeout(pollFinish,1500);}}}}
}}
async function finishReview(){{
  if(!interactive||!lifecycleReady||finishing||reviewFinished)return;
  if(loadingBatch){{notify('Wait for the next batch to finish loading.',true);return;}}
  const wasPaused=autoLoadPaused;autoLoadPaused=true;
  if(!await askConfirmation('Finish Review will wait for queued actions, save the cache, refresh identity profiles once, and run the safety benchmark. Continue?')){{autoLoadPaused=wasPaused;maybeLoadNextBatch();return;}}
  finishState({{status:'running'}});
  try{{
    const response=await fetch('/finish',{{method:'POST'}});const result=await response.json();
    if(!response.ok)notify(result.error||'Could not start Finish Review',true);
    if(!finishState(result))pollFinish();else refreshProgress();
  }}catch(error){{notify(error.message,true);pollFinish();}}
}}
async function initializeLifecycle(){{
  updateLifecycleControls();if(!interactive)return;
  try{{
    const response=await fetch('/finish-status');const result=await response.json();
    if(!response.ok)throw new Error(result.error||'Could not read review status');
    if(finishState(result)){{lifecycleReady=true;updateLifecycleControls();return;}}
    if(finishing){{lifecycleReady=true;pollFinish();return;}}
    if(result.status==='failed')autoLoadPaused=true;
    const batchResponse=await fetch('/batch-status');const batch=await batchResponse.json();
    if(!batchResponse.ok)throw new Error(batch.error||'Could not read batch status');
    queueComplete=batch.status==='completed'&&Boolean(batch.complete);
    if(batch.status==='failed')autoLoadPaused=true;
    lifecycleReady=true;loadingBatch=['queued','running'].includes(batch.status);updateLifecycleControls();
    if(loadingBatch)pollBatchStatus();else{{showCluster();pollJobs();}}
  }}catch(error){{lifecycleReady=false;updateLifecycleControls();notify(error.message+' Reload this page to reconnect.',true);}}
}}

checks().forEach(input=>input.addEventListener('click',event=>{{if(event.shiftKey&&lastChecked){{const visible=checks().filter(item=>!item.disabled&&item.closest('.cluster')===activeCluster());const a=visible.indexOf(lastChecked),b=visible.indexOf(input);if(a>=0&&b>=0)visible.slice(Math.min(a,b),Math.max(a,b)+1).forEach(item=>item.checked=input.checked);}}lastChecked=input;updateSelection();}}));
[search,statusFilter,clusterFilter].forEach(control=>control.addEventListener('input',applyFilters));
document.getElementById('previousCluster').addEventListener('click',()=>moveCluster(-1));document.getElementById('nextCluster').addEventListener('click',()=>moveCluster(1));document.getElementById('skipCluster').addEventListener('click',skipCurrent);document.getElementById('loadNextBatch').addEventListener('click',()=>maybeLoadNextBatch(true));document.getElementById('finishReview').addEventListener('click',finishReview);
document.getElementById('selectVisible').addEventListener('click',()=>{{const cluster=activeCluster();if(!cluster)return;cluster.querySelectorAll('.row-select:not(:disabled)').forEach(input=>input.checked=true);updateSelection();}});document.getElementById('clearSelection').addEventListener('click',()=>{{checks().forEach(input=>input.checked=false);updateSelection();}});
document.querySelectorAll('.select-cluster').forEach(button=>button.addEventListener('click',()=>{{button.closest('.cluster').querySelectorAll('.row-select:not(:disabled)').forEach(input=>input.checked=true);updateSelection();}}));
document.querySelectorAll('[data-density]').forEach(button=>button.addEventListener('click',()=>{{const comfortable=button.dataset.density==='comfortable';document.body.classList.toggle('comfortable',comfortable);document.querySelectorAll('[data-density]').forEach(item=>item.classList.toggle('active',item===button));localStorage.setItem('unknownDensity',button.dataset.density);}}));
const viewer=document.getElementById('viewer');document.querySelectorAll('.preview-button').forEach(button=>button.addEventListener('click',()=>{{document.getElementById('viewerImage').src=button.dataset.full;document.getElementById('viewerName').textContent=button.dataset.name;viewer.showModal();}}));document.getElementById('closeViewer').addEventListener('click',()=>viewer.close());
document.querySelectorAll('button[data-action]').forEach(button=>button.addEventListener('click',()=>{{const cluster=button.closest('.cluster'),item=button.closest('.item'),scope=button.dataset.scope,keys=scope==='cluster'?cluster.dataset.items:item.dataset.item,person=button.dataset.person||(item?.querySelector('input.person')?.value||'');decide(button.dataset.action,keys,person);}}));
document.querySelectorAll('.confirm-custom').forEach(button=>button.addEventListener('click',()=>{{const cluster=button.closest('.cluster'),input=cluster.querySelector('.cluster-person');decide('confirm',cluster.dataset.items,input.value);}}));document.querySelectorAll('.cluster-person').forEach(input=>input.addEventListener('keydown',event=>{{if(event.key==='Enter'){{event.preventDefault();input.closest('.cluster').querySelector('.confirm-custom').click();}}}}));
document.getElementById('confirmSelected').addEventListener('click',()=>decide('confirm',selected().map(input=>input.value),batchPerson.value));document.getElementById('keepSelected').addEventListener('click',()=>decide('keep_unknown',selected().map(input=>input.value)));document.getElementById('ignoreSelected').addEventListener('click',()=>decide('ignore',selected().map(input=>input.value)));useSuggestion.addEventListener('click',()=>{{batchPerson.value=useSuggestion.dataset.person||'';}});
document.addEventListener('keydown',event=>{{
  if(event.metaKey||event.ctrlKey||event.altKey||['INPUT','TEXTAREA','SELECT'].includes(document.activeElement?.tagName)||document.querySelector('dialog[open]'))return;
  const cluster=activeCluster();if(!cluster)return;
  if(['1','2','3'].includes(event.key)){{event.preventDefault();cluster.querySelector(`button[data-shortcut="${{event.key}}"]`)?.click();}}
  else if(event.key.toLocaleLowerCase()==='u'){{event.preventDefault();cluster.querySelector('button[data-action="keep_unknown"]')?.click();}}
  else if(event.key.toLocaleLowerCase()==='j'){{event.preventDefault();cluster.querySelector('button[data-action="move_to_junk"]')?.click();}}
  else if(event.key.toLocaleLowerCase()==='n'){{event.preventDefault();skipCurrent();}}
  else if(event.key==='ArrowRight'){{event.preventDefault();moveCluster(1);}}
  else if(event.key==='ArrowLeft'){{event.preventDefault();moveCluster(-1);}}
}});
search.value=localStorage.getItem('unknownSearch')||'';statusFilter.value=localStorage.getItem('unknownStatus')||'pending';clusterFilter.value=localStorage.getItem('unknownCluster')||'all';document.querySelector(`[data-density="${{localStorage.getItem('unknownDensity')||'compact'}}"]`)?.click();applyFilters();updateSelection();initializeLifecycle();
</script></body></html>"""
