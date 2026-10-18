"""One scheduler per state directory. Separate processes own individual training runs."""
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import psutil
from .config import Config
from .store import Store, ACTIVE
from .gpu import discover
from .datasets import validate


def stop_process(pid, started):
    """Check process birth time before signalling a possibly reused PID."""
    try:
        proc = psutil.Process(pid)
        if abs(proc.create_time() - started) > 0.01:
            return
        # The runner creates a new session. Its data-loader children share its group.
        os.killpg(pid, signal.SIGTERM)
        try:
            proc.wait(timeout=8)
        except psutil.TimeoutExpired:
            pass
        # Terminate remaining data-loader descendants even if their leader exited.
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    except (psutil.NoSuchProcess, ProcessLookupError):
        pass

def stop_child(proc):
    """Popen is the authority for children owned by this live worker."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=5)
    except ProcessLookupError:
        proc.wait(timeout=5)


class Scheduler:
    def __init__(self, config, inventory=discover):
        self.config, self.inventory = config, inventory
        config.prepare()
        self.store = Store(config.state)
        self.children = {}

    def recover(self):
        for job in self.store.list():
            if job["status"] in ACTIVE:
                if job["pid"] and job["pid_started"]:
                    stop_process(job["pid"], job["pid_started"])
                self.store.update(job["id"], status="interrupted", error="Worker restarted. Inspect logs and submit a new job.")

    def tick(self):
        for jid, proc in list(self.children.items()):
            job = self.store.get(jid)
            if job["cancel"]:
                self.store.update(jid, status="cancelling")
                stop_child(proc)
            code = proc.poll()
            if code is not None:
                status = "cancelled" if job["cancel"] else ("completed" if code == 0 else "failed")
                self.store.update(jid, status=status, error=None if code == 0 else f"Process exited with code {code}; see training log")
                self.children.pop(jid)
        cards, error = self.inventory(self.config)
        atomic_json(self.config.state / "worker.json", {"heartbeat": time.time(), "mode": self.config.mode, "gpus": cards, "error": error})
        for card in cards:
            if not card["available"]:
                continue
            # Lock survives supervisor death through pass_fds until child exits.
            gpu_lock = (self.config.state / ("gpu-" + card["uuid"] + ".lock")).open("a")
            try:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                gpu_lock.close()
                continue
            try:
                job = self.store.claim(card["uuid"])
                if job:
                    self.launch(job, gpu_lock)
            finally:
                gpu_lock.close()

    def launch(self, job, gpu_lock):
        directory = self.config.state / "jobs" / job["id"]
        directory.mkdir(exist_ok=True)
        try:
            if job["spec"]["mode"] != self.config.mode:
                raise ValueError("Job mode differs from worker mode; submit to the matching worker")
            validated = validate(self.config.datasets, job["spec"]["dataset"])
            if validated["sha256"] != job["spec"]["dataset_sha256"]:
                raise ValueError("Dataset changed after submission; submit it again")
            atomic_json(directory / "job.json", {**job, "dataset_config": validated["yaml"]})
            env = os.environ.copy()
            env.update(HUB_MODE=self.config.mode, MLFLOW_TRACKING_URI=self.config.tracking_uri,
                       CUDA_VISIBLE_DEVICES=job["gpu"] if self.config.mode == "real" else "",
                       YOLO_CONFIG_DIR=str(directory / "ultralytics"), PYTHONUNBUFFERED="1")
            with (directory / "train.log").open("ab") as log:
                proc = subprocess.Popen([sys.executable, "-m", "hub.runner", str(directory)], env=env,
                                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                        pass_fds=(gpu_lock.fileno(),))
            self.children[job["id"]] = proc
            try:
                identity = psutil.Process(proc.pid).create_time()
            except psutil.NoSuchProcess:
                identity = None
            self.store.update(job["id"], status="running", pid=proc.pid, pid_started=identity)
            # Handshake: runner cannot train until its identity is durably recorded.
            (directory / "start").touch()
        except Exception as exc:
            proc = self.children.pop(job["id"], None)
            if proc:
                stop_child(proc)
            self.store.update(job["id"], status="failed", error=str(exc))

    def shutdown(self):
        for jid, proc in self.children.items():
            job = self.store.get(jid)
            stop_child(proc)
            self.store.update(jid, status="cancelled" if job["cancel"] else "interrupted", error="Worker stopped")
        self.children.clear()


def atomic_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, allow_nan=False))
    tmp.replace(path)


def main():
    config = Config.env()
    config.prepare()
    with (config.state / "scheduler.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("A scheduler is already running for this state directory")
        scheduler = Scheduler(config)
        scheduler.recover()
        stopping = False
        def stop(*_):
            nonlocal stopping
            stopping = True
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            while not stopping:
                scheduler.tick()
                time.sleep(1)
        finally:
            scheduler.shutdown()

if __name__ == "__main__":
    main()
