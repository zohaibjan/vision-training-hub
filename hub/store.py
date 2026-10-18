import json
import sqlite3
import time
import uuid
from contextlib import contextmanager

ACTIVE = ("starting", "running", "cancelling")
TERMINAL = ("completed", "failed", "cancelled", "interrupted")

class Store:
    def __init__(self, root):
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "jobs.db"
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute('''CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, created REAL NOT NULL, updated REAL NOT NULL,
                status TEXT NOT NULL, spec TEXT NOT NULL, gpu TEXT, pid INTEGER,
                pid_started REAL, error TEXT, cancel INTEGER NOT NULL DEFAULT 0)''')
            db.execute("CREATE UNIQUE INDEX IF NOT EXISTS gpu_reservation ON jobs(gpu) WHERE status IN ('starting','running','cancelling')")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=20)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def decode(row):
        if row is None:
            return None
        result = dict(row)
        result["spec"] = json.loads(result["spec"])
        return result

    def add(self, spec):
        job_id, now = uuid.uuid4().hex, time.time()
        with self.connect() as db:
            db.execute("INSERT INTO jobs(id,created,updated,status,spec) VALUES(?,?,?,'queued',?)", (job_id, now, now, json.dumps(spec)))
        return self.get(job_id)

    def get(self, job_id):
        with self.connect() as db:
            return self.decode(db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())

    def list(self):
        with self.connect() as db:
            return [self.decode(r) for r in db.execute("SELECT * FROM jobs ORDER BY created DESC")]

    def claim(self, gpu):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM jobs WHERE gpu=? AND status IN ('starting','running','cancelling')", (gpu,)).fetchone():
                return None
            row = db.execute("SELECT id FROM jobs WHERE status='queued' AND cancel=0 ORDER BY created,id LIMIT 1").fetchone()
            if not row:
                return None
            db.execute("UPDATE jobs SET status='starting',gpu=?,updated=? WHERE id=?", (gpu, time.time(), row["id"]))
            return self.decode(db.execute("SELECT * FROM jobs WHERE id=?", (row["id"],)).fetchone())

    def update(self, job_id, **fields):
        allowed = {"status", "pid", "pid_started", "error", "gpu"}
        if not fields or not fields.keys() <= allowed:
            raise ValueError("Unsupported update")
        fields["updated"] = time.time()
        with self.connect() as db:
            db.execute("UPDATE jobs SET " + ",".join(f"{k}=?" for k in fields) + " WHERE id=?", (*fields.values(), job_id))

    def cancel(self, job_id):
        with self.connect() as db:
            db.execute("UPDATE jobs SET cancel=1,status=CASE WHEN status='queued' THEN 'cancelled' ELSE status END,updated=? WHERE id=? AND status IN ('queued','starting','running','cancelling')", (time.time(), job_id))
