import cv2
import pandas as pd
import numpy as np

# =========================
# 1. File paths
# =========================
csv_file = r"/ultralytics/hara/tools/registrar/.data/20250825T164000Z_20250825T170000Z_mock/20250825T164000Z_20250825T170000Z_RGB.csv"
img1_file = r"/ultralytics/hara/tools/registrar/.data/20250825T164000Z_20250825T170000Z_mock/rgb_frame_00_01_01.png"
img2_file = r"/ultralytics/hara/tools/registrar/.data/20250825T164000Z_20250825T170000Z_mock/thermal_frame_00_01_01.png"
output_file = r"/ultralytics/hara/tools/registrar/.data/20250825T164000Z_20250825T170000Z_mock/comparison_thermal_frame_00_01_01.png"

# =========================
# 2. Read data
# =========================
df = pd.read_csv(csv_file)

# Expected columns: x1, y1, x2, y2
required_cols = {"x1", "y1", "x2", "y2"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"CSV must contain columns: {required_cols}")

# =========================
# 3. Load images
# =========================
img1 = cv2.imread(img1_file)
img2 = cv2.imread(img2_file)

if img1 is None:
    raise FileNotFoundError(f"Cannot read image: {img1_file}")
if img2 is None:
    raise FileNotFoundError(f"Cannot read image: {img2_file}")

# =========================
# 4. Make same height for side-by-side display
# =========================
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

target_h = max(h1, h2)

def resize_to_height(img, target_h):
    h, w = img.shape[:2]
    scale = target_h / h
    new_w = int(w * scale)
    return cv2.resize(img, (new_w, target_h))

img1_resized = resize_to_height(img1, target_h)
img2_resized = resize_to_height(img2, target_h)

h1r, w1r = img1_resized.shape[:2]
h2r, w2r = img2_resized.shape[:2]

# Scale coordinates accordingly
scale1_x = w1r / w1
scale1_y = h1r / h1
scale2_x = w2r / w2
scale2_y = h2r / h2

# =========================
# 5. Create side-by-side canvas
# =========================
canvas = np.zeros((target_h, w1r + w2r, 3), dtype=np.uint8)
canvas[:, :w1r] = img1_resized
canvas[:, w1r:w1r + w2r] = img2_resized

# =========================
# 6. Draw points and lines
# =========================
# Some colors to cycle through
colors = [
    (255, 0, 0),    # Blue
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 128, 255),
    (255, 128, 128),
    (128, 255, 128),
]

for i, row in df.iterrows():
    color = colors[i % len(colors)]

    # Original coordinates
    x1, y1 = row["x1"], row["y1"]
    x2, y2 = row["x2"], row["y2"]

    # Rescaled coordinates
    p1 = (int(x1 * scale1_x), int(y1 * scale1_y))
    p2 = (int(x2 * scale2_x) + w1r, int(y2 * scale2_y))  # shift right image by w1r

    # Draw circles
    cv2.circle(canvas, p1, 6, color, -1)
    cv2.circle(canvas, p2, 6, color, -1)

    # Draw labels
    cv2.putText(canvas, f"{i}", (p1[0] + 8, p1[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{i}", (p2[0] + 8, p2[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # Draw connection line
    cv2.line(canvas, p1, p2, color, 2)

# =========================
# 7. Save and show
# =========================
cv2.imwrite(output_file, canvas)
print(f"Saved result to: {output_file}")

cv2.imshow("Matched Points Visualization", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()