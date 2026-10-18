import json
from pathlib import Path
import secrets
import time
from typing import Literal
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from .config import Config
from .datasets import validate
from .store import Store

class Submission(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(default="YOLOv8 experiment", min_length=1, max_length=100)
    dataset: str = Field(min_length=1, max_length=500)
    model: Literal["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"] = "yolov8n.pt"
    epochs: int = Field(default=50, ge=1, le=1000)
    batch: int = Field(default=8, ge=1, le=256)
    imgsz: int = Field(default=640, ge=32, le=2048, multiple_of=32)
    seed: int = Field(default=42, ge=0, le=2147483647)


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def create_app(config=None):
    config = config or Config.env()
    config.prepare()
    store = Store(config.state)
    app = FastAPI(title="Vision Training Hub", docs_url=None, redoc_url=None)
    def auth(request: Request):
        if not config.token:
            raise HTTPException(503, "Set HUB_API_TOKEN before using the portal")
        if not secrets.compare_digest(request.headers.get("Authorization", ""), "Bearer " + config.token):
            raise HTTPException(401, "Enter the portal access token")
    guarded = [Depends(auth)]
    def get_job(job_id):
        job = store.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        directory = config.state / "jobs" / job_id
        job["progress"] = read_json(directory / "progress.json")
        job["tracking"] = read_json(directory / "tracking.json")
        job["artifacts"] = [str(p.relative_to(directory / "training")) for p in (directory / "training").rglob("*") if p.is_file() and p.suffix in {".pt", ".csv", ".png", ".jpg", ".yaml"}]
        return job

    @app.get("/api/system", dependencies=guarded)
    def system():
        worker = read_json(config.state / "worker.json")
        return {"mode": config.mode, "worker_alive": time.time() - worker.get("heartbeat", 0) < 15,
                "worker": worker, "tracking_uri": config.tracking_uri}

    @app.get("/api/datasets", dependencies=guarded)
    def datasets():
        return {"root": str(config.datasets), "datasets": sorted(str(p.parent.relative_to(config.datasets)) for p in config.datasets.rglob("data.yaml") if p.resolve().is_relative_to(config.datasets))}

    @app.post("/api/jobs", status_code=201, dependencies=guarded)
    def submit(spec: Submission):
        try:
            result = validate(config.datasets, spec.dataset)
        except (ValueError, OSError, TypeError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from exc
        data = spec.model_dump()
        data.update(dataset_sha256=result["sha256"], counts=result["counts"], mode=config.mode)
        return store.add(data)

    @app.get("/api/jobs", dependencies=guarded)
    def jobs():
        return [get_job(j["id"]) for j in store.list()]

    @app.get("/api/jobs/{job_id}", dependencies=guarded)
    def job(job_id: str):
        return get_job(job_id)

    @app.post("/api/jobs/{job_id}/cancel", dependencies=guarded)
    def cancel(job_id: str):
        get_job(job_id)
        store.cancel(job_id)
        return get_job(job_id)

    @app.get("/api/jobs/{job_id}/logs", dependencies=guarded)
    def logs(job_id: str):
        get_job(job_id)
        path = config.state / "jobs" / job_id / "train.log"
        if not path.exists():
            return {"text": "Waiting for worker to start this job."}
        with path.open("rb") as file:
            file.seek(max(0, path.stat().st_size - 64000))
            return {"text": file.read(64000).decode(errors="replace")}

    @app.get("/api/jobs/{job_id}/artifacts/{filename:path}", dependencies=guarded)
    def artifact(job_id: str, filename: str):
        job = get_job(job_id)
        base = (config.state / "jobs" / job_id / "training").resolve()
        path = (base / filename).resolve()
        if not path.is_relative_to(base) or filename not in job["artifacts"]:
            raise HTTPException(404, "Artifact not found")
        return FileResponse(path, filename=path.name)

    app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="portal")
    return app
