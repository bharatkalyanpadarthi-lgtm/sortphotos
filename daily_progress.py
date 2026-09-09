"""Compact presentation of existing worker output; no recognition or file scans."""

from __future__ import annotations

import re
import time
from collections import deque


STEP_LABELS = {
    "preflight": "Checking storage and safety",
    "video-process": "Processing new videos",
    "process": "Processing new photos",
    "structure": "Checking person folders",
    "rename": "Updating photo filenames",
    "exact-dedupe": "Checking exact duplicates (report only)",
    "advanced-dedupe": "Updating duplicate report",
    "cleanup-empty": "Tidying empty folders",
    "cache-rehydrate": "Updating the face cache",
    "unknown-triage": "Preparing the review report",
    "integration-audit": "Checking pipeline consistency",
    "status": "Preparing the final summary",
}
STAGE_LABELS = {
    "Verifying confirmed references": "Checking reference photos",
    "Building trusted recovery profiles": "Preparing known people",
    "Building secondary profiles": "Updating independent face matcher",
    "Calibrating secondary profiles": "Checking independent face matcher",
    "Selecting benchmark samples by person": "Preparing matching safety check",
    "Checking saved benchmark face selections": "Checking saved benchmark selections",
    "Primary-match safety benchmark": "Matching safety check (primary)",
    "Independent-match safety benchmark": "Matching safety check (independent)",
    "Protected benchmark (strict)": "Protected safety check (strict)",
    "Protected benchmark (consensus)": "Protected safety check (consensus)",
    "Protected benchmark (pipeline)": "Protected safety check (filing)",
    "Refreshing identity profiles": "Updating known people",
    "Copying originals": "Saving originals (person groups)",
    "Worker": "Finding faces (current batch)",
}
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
LOG_PREFIX = re.compile(r"^\d{2}:\d{2}:\d{2}\s+(INFO|WARNING|ERROR|CRITICAL|DEBUG)\s+")
COUNTER = re.compile(r"(?<!\d)([\d,]+)\s*/\s*([\d,]+)(?!\d)")


def duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m {seconds:02d}s"


def clean_line(line: str) -> tuple[str, str]:
    text = ANSI.sub("", line).strip()
    match = LOG_PREFIX.match(text)
    level = match.group(1) if match else ""
    if match:
        text = text[match.end():]
    text = "".join(ch for ch in text if ch.isprintable() or ch == "\t")
    return text, level


class CommandProgress:
    def __init__(self, label: str, *, emit=None, clock=time.monotonic, interval=5.0):
        self.label = label
        self.emit = emit or (lambda message: print(message, flush=True))
        self.clock = clock
        self.interval = interval
        self.started = clock()
        self.last_visible = self.started
        self.phase = ""
        self.current = ""
        self.last_message = ""
        self.pending = ""
        self.tail = deque(maxlen=6)
        self.seen_notices: set[str] = set()

    def display(self, message: str) -> None:
        self.emit(f"  {message}")
        self.last_visible = self.clock()
        self.last_message = message

    def update(self, phase: str, detail: str, *, complete=False) -> None:
        changed = phase != self.phase
        self.phase, self.current = phase, detail
        if detail == self.last_message:
            return
        if changed or complete or self.clock() - self.last_visible >= self.interval:
            self.pending = ""
            self.display(detail)
        else:
            self.pending = detail

    def count(self, phase: str, done: int, total: int, *, suffix="") -> None:
        if total <= 0 or not 0 <= done <= total:
            return
        self.update(phase, f"{phase}: {done:,}/{total:,} ({done * 100 // total}%){suffix}", complete=done == total)

    def notice(self, message: str) -> None:
        if message not in self.seen_notices:
            self.seen_notices.add(message)
            self.display(message)

    def consume(self, line: str) -> None:
        try:
            self._consume(line)
        except (ValueError, OverflowError):
            # Malformed progress numbers must not stop the worker. Its original
            # output is already preserved in the full log.
            return

    def _consume(self, line: str) -> None:
        text, level = clean_line(line)
        if not text:
            return
        self.tail.append(text[:600])
        # Errors and warnings are never filtered as ordinary model chatter.
        if level in {"ERROR", "CRITICAL"} or re.match(r"^(?:ERROR\b|\[FAIL\]|Traceback|\w*(?:Error|Exception):)", text):
            self.display(text)
            return
        if level == "WARNING" or re.match(r"^(?:WARNING\b|\[WARN\])", text):
            self.display(f"Warning: {text}")
            return
        if re.match(r"^(?:insufficient_space|low_memory|copy_failed|processing_failed):", text):
            self.display(text)
            return
        if "Safety benchmark: reusing unchanged cached result" in text:
            self.notice("Matching safety check: reusing the saved result.")
            return
        if "Safety benchmark: inputs changed or no cached result" in text:
            self.notice("Matching safety check: inputs changed or no saved result; checking again.")
            return
        if text.startswith(("Safety benchmark blocked", "Safe auto-match is blocked")):
            self.notice("Automatic recovery is paused by the safety check; ambiguous photos remain for review.")
            return
        if text.startswith(("Safe auto-match unavailable:", "Safe auto-match is off because", "Recovery unavailable:", "Verifier cache save failed")):
            self.notice(text)
            return
        if text.startswith("Independent verifier:") and any(word in text.lower() for word in ("unavailable", "failed", "missing", "not built")):
            self.notice(text)
            return
        if text.startswith("Safe auto-match passed "):
            self.notice("Matching safety check passed.")
            return
        if text.startswith("Safety benchmark: checking cached result"):
            self.update("Matching safety check", "Checking whether the saved matching safety result can be reused...")
            return
        if text.startswith("Initializing face detector"):
            self.update("Loading detector", "Loading the face detector...")
            return
        if text.startswith("Indexing moved references for ") or text.startswith("Verifying moved references for "):
            return
        for source, label in STAGE_LABELS.items():
            if text.startswith(source + ":"):
                match = COUNTER.search(text[len(source) + 1:])
                if match:
                    self.count(label, *(int(value.replace(",", "")) for value in match.groups()))
                return
        for prefix, label in (
            ("Intake duplicate check: indexed ", "Checking existing photos for duplicates"),
            ("Intake duplicate check: checked ", "Checking new photos for duplicates"),
            ("Individual identity recovery ", "Checking difficult faces"),
        ):
            if text.startswith(prefix):
                match = COUNTER.search(text[len(prefix):])
                if match:
                    self.count(label, *(int(value.replace(",", "")) for value in match.groups()))
                return
        match = re.match(r"Worker batch (\d+) complete: ([\d,]+) image\(s\) left the inbox; ([\d,]+) remain", text)
        if match:
            batch, handled, remaining = (int(value.replace(",", "")) for value in match.groups())
            self.update("Photo batch complete", f"Batch {batch} complete: {handled:,} photos left the inbox; {remaining:,} remain.", complete=True)
            return
        match = re.match(r"Remaining:\s+([\d,]+) image", text)
        if match:
            self.update("Photo inbox", f"Photo inbox: {int(match[1].replace(',', '')):,} remaining.")
            return
        match = re.match(r"Worker chunk (\d+)/(\d+): (\d+) video", text)
        if match:
            self.update("Video batch", f"Video batch {match[1]}/{match[2]}: {match[3]} videos.")
            return
        match = re.match(r"\[(\d+)/(\d+)\] Analyzing (.+)", text)
        if match:
            self.update("Video file", f"Video {match[1]}/{match[2]} in this batch: {match[3][:120]}")
            return
        match = re.match(r"--- Batch (\d+)/(\d+): images ([\d,]+)[-\u2013]([\d,]+)", text)
        if match:
            self.update("Finding faces", f"Finding faces: batch {match[1]}/{match[2]}, photos {match[3]}-{match[4]}.")
            return
        match = re.match(r"\[(\d+)/(\d+)\] Detecting images (\d+)-(\d+)", text)
        if match:
            self.update("Updating face cache", f"Face cache: batch {match[1]}/{match[2]}, checking photos {match[3]}-{match[4]}.")
            return
        match = re.match(r"Cache hit: (\d+) images \(\d+ faces\). New / changed: (\d+) images", text)
        if match:
            self.update("Cached analysis", f"Reusing face analysis for {int(match[1]):,} photos; {int(match[2]):,} new or changed photos to check.")
            return
        for prefix, label in (
            ("Already cached candidate files:", "Photos already in the face cache"),
            ("Remaining candidate files:", "Photos needing face cache updates"),
            ("Fingerprint cache hits:", "Duplicate checks reused from cache"),
        ):
            if text.startswith(prefix) and text[len(prefix):].strip().isdigit():
                count = int(text[len(prefix):].strip())
                self.update(label, f"{label}: {count:,}.")
                return
        match = re.match(r"(?:Read/signature errors|Decode/read errors):\s+(\d+)", text)
        if match and int(match[1]):
            self.notice(f"Warning: {int(match[1]):,} files could not be read; see the detailed log.")
            return
        if text.startswith("No missing candidate images selected"):
            self.notice("Face cache already current; no new photos need detection.")
            return
        if text.startswith("Preserved ") and "unresolved intake image(s) for review:" in text:
            match = re.match(r"Preserved (\d+) unresolved", text)
            if match:
                self.notice(f"This batch: {int(match[1]):,} unresolved photos retained for review.")
            return
        if text.startswith("Image intake complete:"):
            self.update("Photo inbox complete", "Photo inbox complete: no photos remain in To Process.", complete=True)
        elif text.startswith(("No videos need analysis", "No videos found in:")):
            self.notice("No videos waiting.")
        elif text.startswith("Video analysis complete:"):
            self.update("Video analysis complete", text.split(" across ")[0] + ".", complete=True)

    def tick(self) -> None:
        if self.pending and self.clock() - self.last_visible >= self.interval:
            self.display(self.pending)
            self.pending = ""
        elif self.clock() - self.last_visible >= 30:
            self.display(f"{self.current or self.label} | {duration(self.clock() - self.started)} elapsed in this step; waiting for the next update.")

    def finish(self, returncode: int) -> None:
        if self.pending:
            self.display(self.pending)
            self.pending = ""
        if returncode:
            self.display("The step stopped. Last worker messages:")
            for line in self.tail:
                self.emit(f"    {line}")


class OutputLines:
    """Handle both newline logs and carriage-return progress with bounded memory."""
    def __init__(self, consume, *, limit=65536):
        self.consume = consume
        self.pending = ""
        self.limit = limit

    def feed(self, text: str) -> None:
        parts = re.split(r"[\r\n]", self.pending + text)
        self.pending = parts.pop()
        for part in parts:
            if part:
                self.consume(part)
        while len(self.pending) > self.limit:
            self.consume(self.pending[:self.limit])
            self.pending = self.pending[self.limit:]

    def finish(self) -> None:
        if self.pending:
            self.consume(self.pending)
            self.pending = ""
