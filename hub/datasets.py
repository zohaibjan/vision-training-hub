"""Strict local YOLO detection dataset validation; never execute YAML download hooks."""
import hashlib
import math
from pathlib import Path
import yaml
from PIL import Image

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def contained(path, root):
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("Dataset paths must stay inside the configured dataset root")
    return resolved

def validate(root, selection):
    folder = contained(root / selection, root)
    config_path = contained(folder / "data.yaml", folder)
    if not config_path.is_file():
        raise ValueError("Select a dataset folder containing data.yaml")
    try:
        raw = yaml.safe_load(config_path.read_text())
    except yaml.YAMLError as exc:
        raise ValueError("Invalid data.yaml syntax") from exc
    if not isinstance(raw, dict) or set(raw) - {"path", "train", "val", "names", "nc", "test"}:
        raise ValueError("Unsupported data.yaml keys; download scripts and remote datasets are not allowed")
    names = raw.get("names")
    if isinstance(names, dict):
        if set(names) != set(range(len(names))):
            raise ValueError("Class IDs must start at 0 and be contiguous")
        names = [names[i] for i in range(len(names))]
    if not isinstance(names, list) or not names or not all(isinstance(n, str) and n.strip() for n in names):
        raise ValueError("Provide a nonempty list of class names")
    if "nc" in raw and raw["nc"] != len(names):
        raise ValueError("nc does not match names")
    # Portable datasets use paths relative to this data.yaml, never a different root.
    if raw.get("path", ".") not in (".", "", None):
        raise ValueError("Omit path or use path: .; split folders are relative to data.yaml")
    spec = {"path": str(folder), "names": names}
    counts, digest, seen = {}, hashlib.sha256(), set()
    digest.update(yaml.safe_dump(raw, sort_keys=True).encode())
    for split in ("train", "val"):
        relative = raw.get(split)
        if not isinstance(relative, str) or not relative.startswith("images/"):
            raise ValueError(f"{split} must be a folder such as images/{split}")
        images = contained(folder / relative, folder)
        if not images.is_dir():
            raise ValueError(f"Missing image directory: {relative}")
        files = sorted(p for p in images.rglob("*") if p.suffix.lower() in EXTENSIONS)
        if not files:
            raise ValueError(f"No images in {split}")
        objects = 0
        for image in files:
            image = contained(image, folder)
            if image in seen:
                raise ValueError("Training and validation images overlap")
            seen.add(image)
            with Image.open(image) as opened:
                opened.verify()
            label = contained(folder / "labels" / image.relative_to(folder / "images").with_suffix(".txt"), folder)
            if not label.is_file():
                raise ValueError(f"Missing label: {label.relative_to(folder)}; use an empty file for background images")
            for line in label.read_text().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    raise ValueError("Detection labels require: class x_center y_center width height")
                values = [float(x) for x in parts]
                cls, x, y, w, h = values
                if not all(math.isfinite(v) for v in values) or not cls.is_integer() or not 0 <= cls < len(names) or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    raise ValueError("Invalid class ID or normalized bounding box")
                objects += 1
            for file in (image, label):
                digest.update(str(file.relative_to(folder)).encode())
                with file.open("rb") as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(block)
        if objects == 0:
            raise ValueError(f"{split} must contain at least one labelled object")
        spec[split] = str(images)
        counts[split] = len(files)
    return {"yaml": spec, "counts": counts, "sha256": digest.hexdigest()}
