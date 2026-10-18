"""Make tiny synthetic labelled images; not a meaningful model benchmark."""
from pathlib import Path
from PIL import Image, ImageDraw
root = Path("datasets/synthetic-boxes")
for split, count in (("train", 8), ("val", 4)):
    for i in range(count):
        image = root / "images" / split / f"{i}.png"
        label = root / "labels" / split / f"{i}.txt"
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        canvas = Image.new("RGB", (128, 128), (230, 230-i*5, 220))
        ImageDraw.Draw(canvas).rectangle((32, 32, 96, 96), fill=(40+i*8, 80, 60))
        canvas.save(image)
        label.write_text("0 0.5 0.5 0.5 0.5\n")
(root / "data.yaml").write_text("train: images/train\nval: images/val\nnames: [box]\n")
print(root.resolve())
