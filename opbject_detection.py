from pathlib import Path

import cv2
from ultralytics import YOLO

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODELS_DIR / "yolov8n.pt"
CONFIDENCE = 0.45


def load_model():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    source = str(MODEL_PATH) if MODEL_PATH.exists() else "yolov8n.pt"
    print(f"Loading object detection model ({source})...")
    model = YOLO(source)

    # Cache weights locally if Ultralytics just downloaded them
    if not MODEL_PATH.exists():
        try:
            downloaded = Path(model.ckpt_path)
            if downloaded.exists():
                MODEL_PATH.write_bytes(downloaded.read_bytes())
        except Exception:
            pass

    return model


def main():
    model = load_model()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open front camera (index 0).")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Object detection running. Press Q to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from camera.")
            break

        frame = cv2.flip(frame, 1)
        results = model.predict(frame, conf=CONFIDENCE, verbose=False)

        for result in results:
            names = result.names
            if result.boxes is None:
                continue

            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0])
                score = float(box.conf[0])
                label = f"{names.get(cls_id, cls_id)} {score:.0%}"

                color = (80, 200, 120)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(28, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        cv2.putText(
            frame,
            "Q = quit",
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Object Detection", frame)

        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
