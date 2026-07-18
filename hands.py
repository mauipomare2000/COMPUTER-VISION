import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"

# MediaPipe hand skeleton (21 landmarks)
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)


def draw_hands(frame, detection_result):
    h, w, _ = frame.shape
    for landmarks, handedness in zip(
        detection_result.hand_landmarks,
        detection_result.handedness,
    ):
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, points[a], points[b], (80, 200, 120), 2, cv2.LINE_AA)

        for x, y in points:
            cv2.circle(frame, (x, y), 4, (40, 255, 180), -1, cv2.LINE_AA)

        label = handedness[0].category_name
        score = handedness[0].score
        x0 = min(p[0] for p in points)
        y0 = min(p[1] for p in points)
        cv2.putText(
            frame,
            f"{label} {score:.0%}",
            (x0, max(24, y0 - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (40, 255, 180),
            2,
            cv2.LINE_AA,
        )


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model at {MODEL_PATH}. Download hand_landmarker.task first."
        )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open front camera (index 0).")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        print("Hand detection running. Press Q to quit.")
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from camera.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                draw_hands(frame, result)

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
            cv2.imshow("Hand Detection", frame)

            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
