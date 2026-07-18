import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).resolve().parent / "models" / "pose_landmarker_lite.task"

# MediaPipe pose skeleton (33 landmarks)
POSE_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
)


def draw_pose(frame, detection_result):
    h, w, _ = frame.shape
    for landmarks in detection_result.pose_landmarks:
        points = []
        for lm in landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            visible = lm.visibility > 0.5 if hasattr(lm, "visibility") else True
            points.append((x, y, visible))

        for a, b in POSE_CONNECTIONS:
            if points[a][2] and points[b][2]:
                cv2.line(
                    frame,
                    (points[a][0], points[a][1]),
                    (points[b][0], points[b][1]),
                    (80, 200, 120),
                    2,
                    cv2.LINE_AA,
                )

        for x, y, visible in points:
            if visible:
                cv2.circle(frame, (x, y), 4, (40, 255, 180), -1, cv2.LINE_AA)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model at {MODEL_PATH}. Download pose_landmarker_lite.task first."
        )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open front camera (index 0).")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        print("Pose detection running. Press Q to quit.")
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

            if result.pose_landmarks:
                draw_pose(frame, result)

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
            cv2.imshow("Pose Detection", frame)

            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
