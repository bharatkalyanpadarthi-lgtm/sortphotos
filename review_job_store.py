"""Durable per-job review receipts, independent of the browser connection."""
import json
import sqlite3


class ReviewJobStore:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        self.connection.commit()

    def save(self, job):
        self.save_many([job])

    def save_many(self, jobs):
        with self.connection:
            for job in jobs:
                payload = {key: value for key, value in job.items() if key != "item_key_set"}
                self.connection.execute("INSERT INTO jobs VALUES(?, ?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload",
                                        (job["id"], json.dumps(payload)))

    def load(self):
        jobs = []
        for row in self.connection.execute("SELECT payload FROM jobs"):
            job = json.loads(row[0])
            job["item_key_set"] = frozenset(job["item_keys"])
            if job["status"] in {"queued", "running"}:
                # Never pretend a partly executed request completed. Content-bound
                # decisions remain authoritative when the user retries its items.
                job["status"] = "failed"
                job["error"] = "Interrupted. Saved decisions are preserved; reload these items and retry the unfinished action."
            jobs.append(job)
        return jobs

    def close(self):
        self.connection.close()
