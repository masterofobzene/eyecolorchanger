"""
AI Eye Color Changer – requires deploy.prototxt and res10_300x300_ssd_iter_140000_fp16.caffemodel
from https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt AND 
https://github.com/gopinath-balu/computer_vision/blob/master/CAFFE_DNN/res10_300x300_ssd_iter_140000.caffemodel
"""
import cv2
import mediapipe as mp
import numpy as np
import sys
import os
import shutil
import logging

# ---------- SETUP ----------
DEBUG = True
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EYE_COLORS = {
    "blue":   (180, 120, 40),
    "teal":   (255, 255, 230),
    "green":  (70, 140, 60),
    "brown":  (50, 70, 120),
    "hazel":  (80, 120, 140),
    "gray":   (140, 140, 140),
    "purple": (140, 70, 120),
}

LEFT_IRIS         = [474, 475, 476, 477]
RIGHT_IRIS        = [469, 470, 471, 472]
LEFT_EYE_OUTLINE  = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
RIGHT_EYE_OUTLINE = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.2
)

# DNN face detector
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROTO_TXT = os.path.join(SCRIPT_DIR, r"C:\YOUR_PATH_TO\deploy.prototxt")
CAFFE_MODEL = os.path.join(SCRIPT_DIR, r"C:\YOUR_PATH_TO\res10_300x300_ssd_iter_140000.caffemodel")
face_net = cv2.dnn.readNetFromCaffe(PROTO_TXT, CAFFE_MODEL)

# ---------- FUNCTIONS ----------
def get_landmark_points(landmarks, indices, width, height):
    pts = []
    for idx in indices:
        lm = landmarks.landmark[idx]
        x = int(lm.x * width)
        y = int(lm.y * height)
        pts.append((x, y))
    return np.array(pts, dtype=np.int32)

def create_iris_mask(image, iris_points, eye_mask):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    center = np.mean(iris_points, axis=0)
    cx, cy = int(center[0]), int(center[1])
    distances = [np.linalg.norm(p - center) for p in iris_points]
    iris_radius = int(np.mean(distances) * 0.92)
    cv2.circle(mask, (cx, cy), iris_radius, 255, -1)
    if np.sum(eye_mask) > 0:
        mask = cv2.bitwise_and(mask, eye_mask)
    pupil_radius = int(iris_radius * 0.38)
    cv2.circle(mask, (cx, cy), pupil_radius, 0, -1)
    _, highlights = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(highlights))
    mask = cv2.GaussianBlur(mask, (9, 9), 0)
    return mask

def apply_eye_color(image, mask, target_bgr):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(np.uint8([[target_bgr]]), cv2.COLOR_BGR2LAB)[0,0].astype(np.float32)
    L, A, B = cv2.split(lab)
    mask_f = mask.astype(np.float32) / 255.0
    A = A * (1 - mask_f) + target_lab[1] * mask_f
    B = B * (1 - mask_f) + target_lab[2] * mask_f
    merged = cv2.merge([L, A, B])
    result = cv2.cvtColor(np.clip(merged, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return result

def detect_face_bbox(image):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    if detections.shape[2] == 0:
        return None
    i = np.argmax(detections[0, 0, :, 2])
    confidence = detections[0, 0, i, 2]
    if confidence > 0.5:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x, y, x2, y2) = box.astype("int")
        return (x, y, x2 - x, y2 - y)
    return None

def change_eye_color(image_path, color_name):
    if color_name.lower() not in EYE_COLORS:
        raise ValueError(f"Unsupported color '{color_name}'. Available: {list(EYE_COLORS.keys())}")

    image = cv2.imread(image_path)
    if image is None:
        logger.warning(f"Could not open: {image_path}")
        return False

    orig_h, orig_w = image.shape[:2]

    bbox = detect_face_bbox(image)
    if bbox is None:
        logger.info(f"No face detected: {image_path}")
        return False

    x, y, fw, fh = bbox
    margin = 0.3
    x1 = max(0, int(x - fw * margin))
    y1 = max(0, int(y - fh * margin))
    x2 = min(orig_w, int(x + fw * (1 + margin)))
    y2 = min(orig_h, int(y + fh * (1 + margin)))
    crop = image[y1:y2, x1:x2]

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        logger.info(f"No face landmarks: {image_path}")
        return False

    landmarks = results.multi_face_landmarks[0]
    # Map landmarks back to original image
    for lm in landmarks.landmark:
        lm.x = (lm.x * crop.shape[1] + x1) / orig_w
        lm.y = (lm.y * crop.shape[0] + y1) / orig_h

    h, w = orig_h, orig_w
    left_iris  = get_landmark_points(landmarks, LEFT_IRIS,  w, h)
    right_iris = get_landmark_points(landmarks, RIGHT_IRIS, w, h)
    left_eye_pts  = get_landmark_points(landmarks, LEFT_EYE_OUTLINE,  w, h)
    right_eye_pts = get_landmark_points(landmarks, RIGHT_EYE_OUTLINE, w, h)

    left_eye_mask  = np.zeros((h, w), dtype=np.uint8)
    right_eye_mask = np.zeros((h, w), dtype=np.uint8)
    kernel = np.ones((15, 15), np.uint8)
    if len(left_eye_pts) > 0:
        cv2.fillPoly(left_eye_mask, [left_eye_pts], 255)
        left_eye_mask = cv2.dilate(left_eye_mask, kernel, iterations=3)
    if len(right_eye_pts) > 0:
        cv2.fillPoly(right_eye_mask, [right_eye_pts], 255)
        right_eye_mask = cv2.dilate(right_eye_mask, kernel, iterations=3)

    def choose_mask_for_iris(iris_pts, mask_a, mask_b):
        center = np.mean(iris_pts, axis=0)
        cx, cy = int(center[0]), int(center[1])
        if 0 <= cy < h and 0 <= cx < w:
            if mask_a[cy, cx] > 0:
                return mask_a
            if mask_b[cy, cx] > 0:
                return mask_b
            if np.count_nonzero(mask_a) > 0:
                logger.warning(f"Iris center ({cx},{cy}) outside masks, using mask_a")
                return mask_a
        logger.error(f"Iris center ({cx},{cy}) invalid")
        return None

    left_clip  = choose_mask_for_iris(left_iris,  left_eye_mask,  right_eye_mask)
    right_clip = choose_mask_for_iris(right_iris, right_eye_mask, left_eye_mask)

    left_mask  = create_iris_mask(image, left_iris,  left_clip if left_clip is not None else np.zeros((h,w), dtype=np.uint8))
    right_mask = create_iris_mask(image, right_iris, right_clip if right_clip is not None else np.zeros((h,w), dtype=np.uint8))
    combined_mask = cv2.bitwise_or(left_mask, right_mask)

    if np.count_nonzero(combined_mask) == 0:
        logger.error("Combined mask empty – cannot recolor.")
        return False

    result = apply_eye_color(image, combined_mask, EYE_COLORS[color_name.lower()])

    base, ext = os.path.splitext(image_path)
    output_path = f"{base}_{color_name}{ext}"
    cv2.imwrite(output_path, result)
    logger.info(f"Saved: {output_path}")

    try:
        original_dir = os.path.dirname(image_path)
        ojos_dir = os.path.join(original_dir, "OJOS_ORIGINALES")
        os.makedirs(ojos_dir, exist_ok=True)
        filename = os.path.basename(image_path)
        destination = os.path.join(ojos_dir, filename)
        shutil.move(image_path, destination)
        logger.info(f"Moved original -> {destination}")
    except Exception as move_error:
        logger.warning("Could not move original: %s", move_error)

    return True

def main():
    if len(sys.argv) < 3:
        print("Usage: python eye_color_ai.py image.jpg blue")
        print("or: python eye_color_ai.py img1.jpg img2.jpg teal")
        sys.exit(1)

    color_name = sys.argv[-1].lower()
    image_paths = sys.argv[1:-1]

    if color_name not in EYE_COLORS:
        print(f"Invalid color: {color_name}")
        print(f"Available: {list(EYE_COLORS.keys())}")
        sys.exit(1)

    for image_path in image_paths:
        try:
            logger.info(f"Processing: {image_path}")
            success = change_eye_color(image_path, color_name)
            if not success:
                logger.info(f"Skipped: {image_path}")
        except Exception as e:
            logger.error(f"Failed: {image_path} — {e}")

if __name__ == "__main__":
    main()
