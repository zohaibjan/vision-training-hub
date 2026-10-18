import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import yaml
from .worker import atomic_json


def real_training(directory, job):
    import mlflow
    from ultralytics import YOLO, settings
    settings.update({"mlflow": True, "wandb": False, "comet": False, "clearml": False, "neptune": False})
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "vision-training-hub"
    os.environ["MLFLOW_KEEP_RUN_ACTIVE"] = "true"
    os.environ["MLFLOW_RUN"] = job["id"]
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("vision-training-hub")
    data_file = directory / "data.yaml"
    data_file.write_text(yaml.safe_dump(job["dataset_config"]))
    spec = job["spec"]
    # The outer run supplies a stable ID before training and records failures.
    run = mlflow.start_run(run_name=job["id"])
    atomic_json(directory / "tracking.json", {"run_id": run.info.run_id, "experiment_id": run.info.experiment_id})
    try:
        mlflow.set_tags({"hub.job_id": job["id"], "hub.gpu_uuid": job["gpu"], "hub.dataset_sha256": spec["dataset_sha256"]})
        mlflow.log_artifact(str(data_file), "dataset-config")
        model = YOLO(spec["model"])
        def progress(trainer):
            metrics = {k: float(v) for k, v in trainer.metrics.items() if math.isfinite(float(v))}
            atomic_json(directory / "progress.json", {"epoch": trainer.epoch + 1, "epochs": spec["epochs"], "metrics": metrics, "simulated": False})
        model.add_callback("on_fit_epoch_end", progress)
        model.train(data=str(data_file), epochs=spec["epochs"], imgsz=spec["imgsz"], batch=spec["batch"],
                    device=0, workers=2, project=str(directory), name="training", exist_ok=True,
                    seed=spec["seed"], plots=True, save=True)
        # Explicit artifact upload also checks tracking availability after training.
        mlflow.log_artifacts(str(directory / "training"), "training-output")
    except BaseException as exc:
        mlflow.set_tag("hub.error", str(exc)[:1000])
        mlflow.end_run(status="KILLED" if isinstance(exc, KeyboardInterrupt) else "FAILED")
        raise
    else:
        mlflow.end_run(status="FINISHED")


def main():
    directory = Path(sys.argv[1]).resolve()
    for _ in range(100):
        if (directory / "start").exists():
            break
        time.sleep(0.1)
    else:
        raise RuntimeError("Supervisor did not persist process identity; refusing to start")
    job = json.loads((directory / "job.json").read_text())
    def cancelled(*_):
        raise KeyboardInterrupt("Training cancelled")
    signal.signal(signal.SIGTERM, cancelled)
    if os.getenv("HUB_MODE") == "demo":
        print("SIMULATION ONLY: no model is trained and no MLflow run is created.", flush=True)
        for epoch in range(1, job["spec"]["epochs"] + 1):
            time.sleep(0.4)
            atomic_json(directory / "progress.json", {"epoch": epoch, "epochs": job["spec"]["epochs"], "metrics": {}, "simulated": True})
            print(f"Simulated epoch {epoch}/{job['spec']['epochs']}", flush=True)
        print("Simulation complete. No weights generated.", flush=True)
    else:
        real_training(directory, job)

if __name__ == "__main__":
    main()
