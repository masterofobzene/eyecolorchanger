"""
AI Eye Color Changer
-------------------

Features:
- Realistic AI eye recoloring
- Multi-image support
- Windows shell extension compatible
- Moves successfully processed originals into:
    OJOS_ORIGINALES

Requirements:
    pip install mediapipe==0.10.14
    pip install opencv-python==4.10.0.84
    pip install numpy
"""

import cv2
import mediapipe as mp
import numpy as np
import sys
import os
import shutil

# ------------------------------------------------------------
# COLORS (OpenCV uses BGR)
# ------------------------------------------------------------

EYE_COLORS = {
    "blue": (180, 120, 40),
    "teal": (180, 160, 50),
    "green": (70, 140, 60),
    "brown": (50, 70, 120),
    "hazel": (80, 120, 140),
    "gray": (140, 140, 140),
    "purple": (140, 70, 120),
}

# ------------------------------------------------------------
# MEDIAPIPE IRIS LANDMARKS
# ------------------------------------------------------------

LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def get_landmark_points(landmarks, indices, width, height):

    pts = []

    for idx in indices:

        lm = landmarks.landmark[idx]

        x = int(lm.x * width)
        y = int(lm.y * height)

        pts.append((x, y))

    return np.array(pts, dtype=np.int32)

# ------------------------------------------------------------
# IRIS MASK
# ------------------------------------------------------------

def create_iris_mask(image, iris_points):
    """
    Circular iris mask.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mask = np.zeros(gray.shape, dtype=np.uint8)

    # Iris center
    center = np.mean(iris_points, axis=0)

    cx = int(center[0])
    cy = int(center[1])

    # Radius estimate
    distances = []

    for p in iris_points:

        d = np.linalg.norm(p - center)
        distances.append(d)

    radius = int(np.mean(distances))

    # Slight enlargement
    radius = int(radius * 0.92)

    # Circular iris mask
    cv2.circle(
        mask,
        (cx, cy),
        radius,
        255,
        -1
    )

    # Preserve highlights
    highlights = cv2.threshold(
        gray,
        235,
        255,
        cv2.THRESH_BINARY
    )[1]

    mask = cv2.bitwise_and(
        mask,
        cv2.bitwise_not(highlights)
    )

    # Feather edges
    mask = cv2.GaussianBlur(mask, (11, 11), 0)

    return mask

# ------------------------------------------------------------
# EYE RECOLORING
# ------------------------------------------------------------

def apply_eye_color(image, mask, target_bgr):

    mask_f = mask.astype(np.float32) / 255.0

    # --------------------------------------------------------
    # Stronger recolor on darker iris regions
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    ).astype(np.float32)

    darkness = 1.0 - (gray / 255.0)

    darkness_boost = 0.5 + (darkness * 0.8)

    mask_f = mask_f * darkness_boost

    mask_f = np.clip(mask_f, 0.0, 1.0)

    # --------------------------------------------------------
    # LAB COLOR SPACE
    # --------------------------------------------------------

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    ).astype(np.float32)

    target = np.uint8([[target_bgr]])

    target_lab = cv2.cvtColor(
        target,
        cv2.COLOR_BGR2LAB
    )[0][0].astype(np.float32)

    L, A, B = cv2.split(lab)

    blend = mask_f * 0.72

    # Blend chroma only
    A = A * (1.0 - blend) + target_lab[1] * blend
    B = B * (1.0 - blend) + target_lab[2] * blend

    # Convert back
    L = np.clip(L, 0, 255).astype(np.uint8)
    A = np.clip(A, 0, 255).astype(np.uint8)
    B = np.clip(B, 0, 255).astype(np.uint8)

    merged = cv2.merge([L, A, B])

    recolored = cv2.cvtColor(
        merged,
        cv2.COLOR_LAB2BGR
    )

    # --------------------------------------------------------
    # Slight brightness boost
    # --------------------------------------------------------

    brightness_boost = np.zeros_like(recolored)

    brightness_boost[:, :] = (12, 8, 0)

    recolored = cv2.add(
        recolored,
        brightness_boost
    )

    # --------------------------------------------------------
    # Final blend
    # --------------------------------------------------------

    result = image.astype(np.float32)

    for c in range(3):

        result[:, :, c] = (
            image[:, :, c] * (1.0 - blend * 0.9)
            + recolored[:, :, c] * (blend * 0.9)
        )

    result = np.clip(
        result,
        0,
        255
    ).astype(np.uint8)

    return result

# ------------------------------------------------------------
# PROCESS IMAGE
# ------------------------------------------------------------

def change_eye_color(image_path, color_name):

    if color_name.lower() not in EYE_COLORS:

        raise ValueError(
            f"Unsupported color '{color_name}'. "
            f"Available: {list(EYE_COLORS.keys())}"
        )

    image = cv2.imread(image_path)

    if image is None:

        print(f"Could not open: {image_path}")
        return False

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:

        print(f"No face detected: {image_path}")
        return False

    landmarks = results.multi_face_landmarks[0]

    h, w = image.shape[:2]

    # Left iris
    left_iris = get_landmark_points(
        landmarks,
        LEFT_IRIS,
        w,
        h
    )

    # Right iris
    right_iris = get_landmark_points(
        landmarks,
        RIGHT_IRIS,
        w,
        h
    )

    # Masks
    left_mask = create_iris_mask(
        image,
        left_iris
    )

    right_mask = create_iris_mask(
        image,
        right_iris
    )

    combined_mask = cv2.bitwise_or(
        left_mask,
        right_mask
    )

    # Apply recolor
    result = apply_eye_color(
        image,
        combined_mask,
        EYE_COLORS[color_name.lower()]
    )

    # --------------------------------------------------------
    # SAVE OUTPUT
    # --------------------------------------------------------

    base, ext = os.path.splitext(image_path)

    output_path = f"{base}_{color_name}{ext}"

    cv2.imwrite(output_path, result)

    print(f"Saved: {output_path}")

    # --------------------------------------------------------
    # MOVE ORIGINAL
    # --------------------------------------------------------

    try:

        original_dir = os.path.dirname(image_path)

        ojos_dir = os.path.join(
            original_dir,
            "OJOS_ORIGINALES"
        )

        os.makedirs(
            ojos_dir,
            exist_ok=True
        )

        filename = os.path.basename(image_path)

        destination = os.path.join(
            ojos_dir,
            filename
        )

        shutil.move(
            image_path,
            destination
        )

        print(f"Moved original -> {destination}")

    except Exception as move_error:

        print("Could not move original:")
        print(move_error)

    return True

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    if len(sys.argv) < 3:

        print("Usage:")
        print("python eye_color_ai.py image.jpg blue")
        print("python eye_color_ai.py image1.jpg image2.jpg teal")

        sys.exit(1)

    # Last argument is color
    color_name = sys.argv[-1].lower()

    # Everything before is image paths
    image_paths = sys.argv[1:-1]

    if color_name not in EYE_COLORS:

        print(f"Invalid color: {color_name}")
        print(f"Available: {list(EYE_COLORS.keys())}")

        sys.exit(1)

    # Process images
    for image_path in image_paths:

        try:

            print(f"\nProcessing: {image_path}")

            success = change_eye_color(
                image_path,
                color_name
            )

            if not success:
                print(f"Skipped: {image_path}")

        except Exception as e:

            print(f"Failed: {image_path}")
            print(f"Error: {e}")

# ------------------------------------------------------------
# ENTRY
# ------------------------------------------------------------

if __name__ == "__main__":
    main()