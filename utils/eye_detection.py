import time
import numpy as np
import cv2
from scipy.spatial import distance as dist

LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]


def calculate_ear(eye_landmarks: np.ndarray) -> float:
    A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)


def extract_eye_landmarks(face_landmarks, image_width: int, image_height: int):
    def get_coords(indices):
        points = []
        for idx in indices:
            lm = face_landmarks.landmark[idx]
            x = int(lm.x * image_width)
            y = int(lm.y * image_height)
            points.append((x, y))
        return np.array(points, dtype=np.float64)
    return get_coords(LEFT_EYE_INDICES), get_coords(RIGHT_EYE_INDICES)


def get_eye_roi(frame: np.ndarray, eye_landmarks: np.ndarray, padding: int = 10):
    x_coords = eye_landmarks[:, 0].astype(int)
    y_coords = eye_landmarks[:, 1].astype(int)
    x_min = max(0, x_coords.min() - padding)
    x_max = min(frame.shape[1], x_coords.max() + padding)
    y_min = max(0, y_coords.min() - padding)
    y_max = min(frame.shape[0], y_coords.max() + padding)
    if x_max <= x_min or y_max <= y_min:
        return None
    roi = frame[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        return None
    return roi


def preprocess_eye_image(roi: np.ndarray, model_name: str) -> np.ndarray:
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    target_size = (96, 96) if model_name == "EfficientNetB0" else (64, 32)
    resized = cv2.resize(roi_rgb, target_size)
    img = resized.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)


def run_eye_model_inference(model, roi: np.ndarray, model_name: str) -> dict:
    start = time.perf_counter()
    input_tensor = preprocess_eye_image(roi, model_name)
    prediction = model.predict(input_tensor, verbose=0)[0]
    confidence = float(prediction[1])
    label = "Strained" if confidence > 0.70 else "Normal"
    if label == "Normal":
        confidence = float(prediction[0])
    latency_ms = (time.perf_counter() - start) * 1000.0
    return {"label": label, "confidence": confidence, "latency_ms": latency_ms}


def draw_eye_landmarks(frame: np.ndarray, left_eye: np.ndarray, right_eye: np.ndarray):
    for eye_pts in [left_eye, right_eye]:
        hull = cv2.convexHull(eye_pts.astype(np.int32))
        cv2.drawContours(frame, [hull], -1, (0, 255, 0), 1)
