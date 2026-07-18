import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).resolve().parent / "models" / "blaze_face_short_range.tflite"


def draw_faces(frame, detection_result):
    h, w, _ = frame.shape
    for detection in detection_result.detections:
        box = detection.bounding_box
        x1 = int(box.origin_x)
        y1 = int(box.origin_y)
        x2 = int(box.origin_x + box.width)
        y2 = int(box.origin_y + box.height)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 200, 120), 2, cv2.LINE_AA)

        score = detection.categories[0].score if detection.categories else 0.0
        cv2.putText(
            frame,
            f"Face {score:.0%}",
            (x1, max(24, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (40, 255, 180),
            2,
            cv2.LINE_AA,
        )

        for keypoint in detection.keypoints:
            kx = int(keypoint.x * w)
            ky = int(keypoint.y * h)
            cv2.circle(frame, (kx, ky), 3, (40, 255, 180), -1, cv2.LINE_AA)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model at {MODEL_PATH}. Download blaze_face_short_range.tflite first."
        )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open front camera (index 0).")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    options = vision.FaceDetectorOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        min_detection_confidence=0.5,
    )

    with vision.FaceDetector.create_from_options(options) as detector:
        print("Face detection running. Press Q to quit.")
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from camera.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.time() * 1000)
            result = detector.detect_for_video(mp_image, timestamp_ms)

            if result.detections:
                draw_faces(frame, result)

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
            cv2.imshow("Face Detection", frame)

            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
