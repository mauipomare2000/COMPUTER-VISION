"""
Train a license-plate detector and run it on a video.

Reads each plate with OCR, counts unique plate values, and writes a report.

Just run:
  python custom/plates.py

Optional:
  python custom/plates.py --video "path/to/video.mp4"
  python custom/plates.py --retrain
  python custom/plates.py --epochs 20
"""

from __future__ import annotations

import argparse
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import easyocr
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset" / "plates" / "License-Plate-Data"
DATA_YAML = DATASET_DIR / "data.yaml"
DEFAULT_VIDEO = ROOT / "media" / "License Plate Detection Test - (1080p).mp4"
RUNS_DIR = ROOT / "runs" / "plates"
WEIGHTS_DIR = ROOT / "models"
BEST_WEIGHTS = WEIGHTS_DIR / "license_plate_best.pt"
OUTPUT_DIR = ROOT / "output"

PLATE_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


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


def normalize_plate(text: str) -> str:
    cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
    return cleaned


def read_plate_text(reader: easyocr.Reader, crop_bgr: np.ndarray) -> tuple[str, float]:
    if crop_bgr is None or crop_bgr.size == 0:
        return "", 0.0

    h, w = crop_bgr.shape[:2]
    if h < 12 or w < 12:
        return "", 0.0

    # Upscale small crops so OCR is more reliable
    scale = max(1.0, 120 / max(h, 1))
    if scale > 1.0:
        crop_bgr = cv2.resize(
            crop_bgr,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_CUBIC,
        )

    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    results = reader.readtext(
        rgb,
        allowlist=PLATE_CHARS,
        detail=1,
        paragraph=False,
    )
    if not results:
        return "", 0.0

    # Join fragments left-to-right
    results = sorted(results, key=lambda r: r[0][0][0])
    texts = []
    scores = []
    for _, text, score in results:
        value = normalize_plate(text)
        if value:
            texts.append(value)
            scores.append(float(score))

    if not texts:
        return "", 0.0

    plate = "".join(texts)
    conf = sum(scores) / len(scores)
    return plate, conf


def write_report(
    report_path: Path,
    video: Path,
    detections: list[dict],
    unique_counts: Counter,
) -> None:
    lines = [
        "License Plate Detection Report",
        "=" * 40,
        f"Video: {video}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Summary",
        "-" * 40,
        f"Total plate detections: {len(detections)}",
        f"Unique plate values: {len(unique_counts)}",
        "",
        "Unique plates",
        "-" * 40,
    ]

    if unique_counts:
        for i, (plate, count) in enumerate(unique_counts.most_common(), start=1):
            lines.append(f"{i}. {plate}  (seen {count} time{'s' if count != 1 else ''})")
    else:
        lines.append("(none)")

    lines.extend(
        [
            "",
            "Detection log",
            "-" * 40,
            "Each row is one detected plate box in a frame.",
            "",
        ]
    )

    if detections:
        for d in detections:
            pid = d["plate_id"] if d["value"] else "-"
            lines.append(
                f"Frame {d['frame']:>6} | Plate #{str(pid):<3} | "
                f"Value: {d['value'] or 'UNREADABLE':<12} | "
                f"OCR conf: {d['ocr_conf']:.0%} | "
                f"Det conf: {d['det_conf']:.0%} | "
                f"Box: ({d['x1']},{d['y1']})-({d['x2']},{d['y2']})"
            )
    else:
        lines.append("(no plates detected)")

    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def detect(
    video: Path,
    weights: Path,
    conf: float = 0.25,
    output: Path | None = None,
    report: Path | None = None,
) -> tuple[Path, Path]:
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = output or (OUTPUT_DIR / f"{video.stem}_plates.mp4")
    report_path = report or (OUTPUT_DIR / f"{video.stem}_plates_report.txt")

    print(f"Loading detector: {weights}")
    model = YOLO(str(weights))

    print("Loading OCR reader (first run may download models)...")
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {out_path}")

    print(f"Running detection + OCR on: {video}")
    print(f"Writing annotated video -> {out_path}")
    print(f"Writing plate report   -> {report_path}")

    detections: list[dict] = []
    unique_counts: Counter = Counter()
    plate_ids: dict[str, int] = {}
    next_plate_id = 1
    frame_i = 0

    # OCR is expensive; reuse last reading for similar boxes briefly
    last_ocr: dict[tuple[int, int, int, int], tuple[str, float]] = {}

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_i += 1

        results = model.predict(frame, conf=conf, verbose=False)
        annotated = frame.copy()

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width - 1, x2), min(height - 1, y2)
                det_conf = float(box.conf[0])

                # Quantize box for light OCR caching across nearby frames
                cache_key = (x1 // 8, y1 // 8, x2 // 8, y2 // 8)
                if cache_key in last_ocr and frame_i % 3 != 1:
                    plate_text, ocr_conf = last_ocr[cache_key]
                else:
                    crop = frame[y1:y2, x1:x2]
                    plate_text, ocr_conf = read_plate_text(reader, crop)
                    last_ocr[cache_key] = (plate_text, ocr_conf)

                if plate_text:
                    if plate_text not in plate_ids:
                        plate_ids[plate_text] = next_plate_id
                        next_plate_id += 1
                    plate_id = plate_ids[plate_text]
                    unique_counts[plate_text] += 1
                    label = f"#{plate_id} {plate_text}"
                else:
                    plate_id = 0
                    label = "UNREADABLE"

                detections.append(
                    {
                        "frame": frame_i,
                        "plate_id": plate_id if plate_text else 0,
                        "value": plate_text,
                        "ocr_conf": ocr_conf,
                        "det_conf": det_conf,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    }
                )

                color = (80, 200, 120) if plate_text else (60, 60, 220)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
                cv2.putText(
                    annotated,
                    label,
                    (x1, max(28, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        # Keep OCR cache from growing forever
        if len(last_ocr) > 200:
            last_ocr.clear()

        writer.write(annotated)

        if frame_i % 30 == 0:
            print(f"  processed {frame_i} frames | unique plates so far: {len(unique_counts)}")

    cap.release()
    writer.release()

    write_report(report_path, video, detections, unique_counts)

    print(f"Done. Annotated video: {out_path}")
    print(f"Done. Plate report:    {report_path}")
    print(f"Unique plates found:   {len(unique_counts)}")
    return out_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train (if needed) and run license-plate detection + OCR on a video."
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
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Output report path (default: custom/output/<video>_plates_report.txt)",
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

    detect(
        video=args.video,
        weights=weights,
        conf=args.conf,
        output=args.output,
        report=args.report,
    )


if __name__ == "__main__":
    main()
