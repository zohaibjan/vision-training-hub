import os
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    state: Path
    datasets: Path
    mode: str = "real"
    token: str = ""
    tracking_uri: str = "http://127.0.0.1:5000"
    min_free_mb: int = 4096
    max_util: int = 10

    @classmethod
    def env(cls):
        mode = os.getenv("HUB_MODE", "real")
        if mode not in {"real", "demo"}:
            raise ValueError("HUB_MODE must be real or demo")
        return cls(Path(os.getenv("HUB_STATE_DIR", "state")).resolve(),
                   Path(os.getenv("HUB_DATASETS_DIR", "datasets")).resolve(), mode,
                   os.getenv("HUB_API_TOKEN", ""),
                   os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
                   int(os.getenv("HUB_MIN_FREE_MB", "4096")),
                   int(os.getenv("HUB_MAX_GPU_UTIL", "10")))

    def prepare(self):
        self.state.mkdir(parents=True, exist_ok=True)
        self.datasets.mkdir(parents=True, exist_ok=True)
        (self.state / "jobs").mkdir(exist_ok=True)
