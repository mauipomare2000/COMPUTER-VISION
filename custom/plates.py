"""
Train a license-plate detector and run it on a video.

Just run:
  python custom/plates.py

Optional:
  python custom/plates.py --video "path/to/video.mp4"
  python custom/plates.py --retrain
  python custom/plates.py --epochs 20
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset" / "plates" / "License-Plate-Data"
DATA_YAML = DATASET_DIR / "data.yaml"
DEFAULT_VIDEO = ROOT / "media" / "License Plate Detection Test - (1080p).mp4"
RUNS_DIR = ROOT / "runs" / "plates"
WEIGHTS_DIR = ROOT / "models"
BEST_WEIGHTS = WEIGHTS_DIR / "license_plate_best.pt"
OUTPUT_DIR = ROOT / "output"


def ensure_data_yaml() -> Path:
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_DIR}")

    DATA_YAML.write_text(
        "\n".join(
            [
                f"path: {DATASET_DIR.as_posix()}",
                "train: train/images",
                "val: test/images",
                "test: test/images",
                "",
                "nc: 1",
                "names:",
                "  0: license_plate",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return DATA_YAML


def find_existing_weights() -> Path | None:
    if BEST_WEIGHTS.exists():
        return BEST_WEIGHTS

    candidates = sorted(
        RUNS_DIR.glob("**/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def train(
    epochs: int = 20,
    imgsz: int = 640,
    batch: int = 8,
    model_name: str = "yolov8n.pt",
) -> Path:
    ensure_data_yaml()
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Training license plate model on {DATASET_DIR}")
    print(f"  epochs={epochs}  imgsz={imgsz}  batch={batch}  base={model_name}")

    model = YOLO(model_name)
    results = model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(RUNS_DIR),
        name="train",
        exist_ok=True,
        patience=15,
        verbose=True,
    )

    run_dir = Path(results.save_dir)
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"Training finished but best weights missing: {best}")

    shutil.copy2(best, BEST_WEIGHTS)
    print(f"Saved trained model -> {BEST_WEIGHTS}")
    return BEST_WEIGHTS


def detect(
    video: Path,
    weights: Path,
    conf: float = 0.25,
    output: Path | None = None,
) -> Path:
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = output or (OUTPUT_DIR / f"{video.stem}_plates.mp4")

    print(f"Loading model: {weights}")
    model = YOLO(str(weights))

    print(f"Running detection on: {video}")
    print(f"Writing annotated video -> {out_path}")

    model.predict(
        source=str(video),
        conf=conf,
        save=True,
        project=str(OUTPUT_DIR),
        name="predict",
        exist_ok=True,
        stream=False,
        verbose=True,
    )

    predict_dir = OUTPUT_DIR / "predict"
    videos = [
        p
        for p in predict_dir.rglob("*")
        if p.suffix.lower() in {".mp4", ".avi", ".mkv"}
    ]
    if not videos:
        raise RuntimeError(f"Detection finished but no output video in {predict_dir}")

    saved = max(videos, key=lambda p: p.stat().st_mtime)
    if saved.resolve() != out_path.resolve():
        if out_path.exists():
            out_path.unlink()
        shutil.copy2(saved, out_path)

    print(f"Done. Annotated video: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train (if needed) and run license-plate detection on a video."
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=DEFAULT_VIDEO,
        help="Input video path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output video path (default: custom/output/<video>_plates.mp4)",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--model", default="yolov8n.pt", help="Base YOLO weights for training")
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force training even if a model already exists",
    )
    args = parser.parse_args()

    weights = find_existing_weights()
    if args.retrain or weights is None:
        weights = train(
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            model_name=args.model,
        )
    else:
        if weights != BEST_WEIGHTS:
            WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(weights, BEST_WEIGHTS)
            weights = BEST_WEIGHTS
        print(f"Using existing model: {weights}")

    detect(video=args.video, weights=weights, conf=args.conf, output=args.output)


if __name__ == "__main__":
    main()
