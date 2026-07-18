import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).resolve().parent / "models" / "blaze_face_short_range.tflite"
EMOTION_EVERY_N_FRAMES = 5
EMOTIONS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")

# BGR colors per emotion
EMOTION_COLORS = {
    "angry": (60, 60, 220),
    "disgust": (80, 160, 80),
    "fear": (180, 80, 180),
    "happy": (40, 200, 80),
    "sad": (200, 140, 40),
    "surprise": (40, 200, 220),
    "neutral": (180, 180, 180),
}


def clamp_box(x1, y1, x2, y2, w, h, pad=0.15):
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * pad))
    y1 = max(0, int(y1 - bh * pad))
    x2 = min(w, int(x2 + bw * pad))
    y2 = min(h, int(y2 + bh * pad))
    return x1, y1, x2, y2


def predict_emotion(face_bgr):
    if face_bgr.size == 0 or face_bgr.shape[0] < 20 or face_bgr.shape[1] < 20:
        return None, {}

    analysis = DeepFace.analyze(
        face_bgr,
        actions=["emotion"],
        enforce_detection=False,
        detector_backend="skip",
        silent=True,
    )
    if isinstance(analysis, list):
        analysis = analysis[0]

    return analysis.get("dominant_emotion"), analysis.get("emotion", {})


def draw_emotion(frame, box, emotion, scores):
    x1, y1, x2, y2 = box
    color = EMOTION_COLORS.get(emotion, (40, 255, 180))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    label = emotion.capitalize() if emotion else "..."
    if emotion and scores:
        label = f"{label} {scores.get(emotion, 0):.0f}%"

    cv2.putText(
        frame,
        label,
        (x1, max(28, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )

    # Small score bars
    if scores:
        bar_x = x2 + 8
        bar_y = y1
        for i, name in enumerate(EMOTIONS):
            value = float(scores.get(name, 0))
            y = bar_y + i * 18
            cv2.putText(
                frame,
                name[:3],
                (bar_x, y + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
            cv2.rectangle(
                frame,
                (bar_x + 28, y + 2),
                (bar_x + 28 + int(value * 0.8), y + 12),
                EMOTION_COLORS[name],
                -1,
                cv2.LINE_AA,
            )


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model at {MODEL_PATH}. Download blaze_face_short_range.tflite first."
        )

    # Warm up emotion model once so the first live frame is faster
    print("Loading emotion model...")
    blank = np.zeros((48, 48, 3), dtype=np.uint8)
    try:
        DeepFace.analyze(
            blank,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="skip",
            silent=True,
        )
    except Exception:
        pass
    print("Ready. Press Q to quit.")

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

    cached = {}  # face_index -> (emotion, scores)
    frame_i = 0

    with vision.FaceDetector.create_from_options(options) as detector:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from camera.")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.time() * 1000)
            result = detector.detect_for_video(mp_image, timestamp_ms)

            run_emotion = frame_i % EMOTION_EVERY_N_FRAMES == 0
            frame_i += 1

            if result.detections:
                for idx, detection in enumerate(result.detections):
                    box = detection.bounding_box
                    x1, y1, x2, y2 = clamp_box(
                        box.origin_x,
                        box.origin_y,
                        box.origin_x + box.width,
                        box.origin_y + box.height,
                        w,
                        h,
                    )

                    if run_emotion:
                        face = frame[y1:y2, x1:x2]
                        try:
                            emotion, scores = predict_emotion(face)
                            cached[idx] = (emotion, scores)
                        except Exception:
                            cached.setdefault(idx, (None, {}))

                    emotion, scores = cached.get(idx, (None, {}))
                    draw_emotion(frame, (x1, y1, x2, y2), emotion, scores)

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
            cv2.imshow("Facial Emotions", frame)

            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
