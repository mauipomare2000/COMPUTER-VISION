"""
Live iPhone stream with YOLO penny detection.

  python Droid/droid.py

Trains automatically if Droid/models/penny_best.pt is missing.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_penny import BEST_WEIGHTS, train

CONFIG_PATH = ROOT / "config.yaml"
CONF_THRESHOLD = 0.25


class VideoStream:
    def __init__(self, url: str):
        self.cap = cv2.VideoCapture(url)
        self.frame = None
        self.running = True
        threading.Thread(target=self.update, daemon=True).start()

    def update(self) -> None:
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame

    def read(self):
        return self.frame

    def release(self) -> None:
        self.running = False
        self.cap.release()


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_model() -> YOLO:
    if not BEST_WEIGHTS.exists():
        print(f"No weights at {BEST_WEIGHTS} — training first...")
        train()
    print(f"Loading model: {BEST_WEIGHTS}")
    return YOLO(str(BEST_WEIGHTS))


def main() -> None:
    config = load_config()
    url = config["url"]
    print(f"Stream: {url}")

    model = load_model()
    stream = VideoStream(url)

    try:
        while True:
            frame = stream.read()
            if frame is None:
                continue

            results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)
            annotated = results[0].plot()

            cv2.imshow("Penny Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        stream.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
