import cv2
import numpy as np

def extract_handwriting_features(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Could not load image")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Invert if background is white
    if np.mean(gray) > 127:
        gray = 255 - gray

    # Apply adaptive threshold
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2
    )

    # Dilate to make strokes thicker
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)

    # Clean noise
    binary = cv2.medianBlur(binary, 3)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No contours found")
    
    letter_sizes = []
    word_gaps = []
    slants = []
    loops = 0
    baseline_y = []
    smoothness = []

    prev_x_right = None
    prev_y_mid = None

    sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])  # sort left to right

    for cnt in sorted_contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Skip very small noise
        if h < 5 or w < 3:
            continue

        # Calculate slant
        angle = 0
        if len(cnt) >= 5:
            try:
                (_, _), (_, _), angle = cv2.fitEllipse(cnt)
            except:
                angle = 0
        slants.append(angle)

        # Word gap (same line)
        y_mid = y + h // 2
        if prev_x_right is not None and abs(y_mid - prev_y_mid) < 20:  # same line
            gap = x - prev_x_right
            if 2 < gap < 150:
                word_gaps.append(gap)
        prev_x_right = x + w
        prev_y_mid = y_mid

        # Letter size
        letter_sizes.append(h)

        # Loop detection
        if h > 2 * w:
            loops += 1

        # Stroke smoothness
        perimeter = cv2.arcLength(cnt, True)
        denom = 2 * (w + h)
        if denom > 0:
            smoothness.append(perimeter / denom)

        baseline_y.append(y_mid)

    # Print debug info
    print("word_gaps:", word_gaps)
    print("letter_sizes:", letter_sizes)
    print("slants:", slants)
    print("loops:", loops)
    print("baseline_y:", baseline_y)
    print("smoothness:", smoothness)

    # Feature calculations
    word_spacing_var = np.std(word_gaps) if len(word_gaps) > 1 else 0
    letter_size_std = np.std(letter_sizes) if len(letter_sizes) > 1 else 0
    legibility = np.sum(binary == 255) / (binary.shape[0] * binary.shape[1])
    baseline_std = np.std(baseline_y) if len(baseline_y) > 1 else 0
    avg_slant = np.mean(slants) if slants else 0
    loop_ratio = loops / len(letter_sizes) if letter_sizes else 0
    avg_smoothness = np.mean(smoothness) if smoothness else 0
    cursive_tendency = np.mean(word_gaps) if word_gaps else 0

    # Final feature vector
    features = np.array([
        avg_slant / 90,                 # slant_angle
        letter_size_std / 100,         # letter_spacing
        baseline_std / 100,            # line_alignment
        loop_ratio,                    # loop_formation
        avg_smoothness,                # stroke_smoothness
        word_spacing_var / 100,        # word_spacing
        letter_size_std / 100,         # letter_size
        legibility,                    # legibility
        1 if cursive_tendency < 20 else 0  # cursive_print
    ])

    return [float(x) if np.isfinite(x) else 0.0 for x in features]
