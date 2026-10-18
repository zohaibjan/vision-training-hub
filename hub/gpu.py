import csv
import io
import subprocess


def discover(config):
    if config.mode == "demo":
        return [{"uuid": f"demo-{n}", "name": "Simulated GPU", "free_mb": 8192, "util": 0, "available": True} for n in range(2)], None
    try:
        def query(args):
            return subprocess.check_output(["nvidia-smi", *args, "--format=csv,noheader,nounits"], text=True, timeout=5)
        rows = query(["--query-gpu=uuid,name,memory.free,utilization.gpu,mig.mode.current"])
        processes = query(["--query-compute-apps=gpu_uuid,pid"])
        busy = {r[0].strip() for r in csv.reader(io.StringIO(processes)) if r}
        cards = []
        for row in csv.reader(io.StringIO(rows)):
            uid, name, free, util, mig = [v.strip() for v in row]
            free, util = int(free), int(util)
            cards.append({"uuid": uid, "name": name, "free_mb": free, "util": util,
                          "available": uid not in busy and free >= config.min_free_mb and util <= config.max_util and mig != "Enabled"})
        return cards, None
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return [], f"GPU discovery unavailable; jobs remain queued ({type(exc).__name__})"
