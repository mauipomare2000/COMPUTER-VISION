"""
Prepare Droid/DATA and train a YOLO penny detector.

  python Droid/train_penny.py
  python Droid/train_penny.py --epochs 80
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "DATA"
DATASET_DIR = ROOT / "dataset" / "penny"
DATA_YAML = DATASET_DIR / "data.yaml"
WEIGHTS_DIR = ROOT / "models"
BEST_WEIGHTS = WEIGHTS_DIR / "penny_best.pt"
RUNS_DIR = ROOT / "runs" / "penny"


def prepare_dataset(val_ratio: float = 0.25, seed: int = 42) -> Path:
    images_dir = RAW_DIR / "images"
    labels_dir = RAW_DIR / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"Expected images/ and labels/ under {RAW_DIR}")

    pairs: list[tuple[Path, Path]] = []
    for img in sorted(images_dir.glob("*")):
        if img.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            continue
        label = labels_dir / f"{img.stem}.txt"
        if label.exists():
            pairs.append((img, label))

    if not pairs:
        raise FileNotFoundError(f"No labeled images found in {RAW_DIR}")

    rng = random.Random(seed)
    rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_ratio))
    if len(pairs) - n_val < 1:
        n_val = 1
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

    for split, split_pairs in (("train", train_pairs), ("val", val_pairs)):
        (DATASET_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / split / "labels").mkdir(parents=True, exist_ok=True)
        for img, label in split_pairs:
            shutil.copy2(img, DATASET_DIR / split / "images" / img.name)
            shutil.copy2(label, DATASET_DIR / split / "labels" / label.name)

    DATA_YAML.write_text(
        "\n".join(
            [
                f"path: {DATASET_DIR.as_posix()}",
                "train: train/images",
                "val: val/images",
                "",
                "nc: 1",
                "names:",
                "  0: penny",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Dataset ready: {len(train_pairs)} train / {len(val_pairs)} val -> {DATASET_DIR}")
    return DATA_YAML


def train(
    epochs: int = 80,
    imgsz: int = 640,
    batch: int = 4,
    model_name: str = "yolov8n.pt",
) -> Path:
    prepare_dataset()
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Training penny model  epochs={epochs} imgsz={imgsz} batch={batch}")
    model = YOLO(model_name)
    results = model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(RUNS_DIR),
        name="train",
        exist_ok=True,
        patience=30,
        verbose=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"Training finished but best weights missing: {best}")

    shutil.copy2(best, BEST_WEIGHTS)
    print(f"Saved trained model -> {BEST_WEIGHTS}")
    return BEST_WEIGHTS


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO penny detector")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--model", default="yolov8n.pt")
    args = parser.parse_args()
    train(epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, model_name=args.model)


if __name__ == "__main__":
    main()
