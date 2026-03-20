import cv2
import numpy as np

# cap = cv2.VideoCapture(r"I:\20250825T220000Z_20250825T222000Z.mkv")
cap = cv2.VideoCapture(r"I:\output.mkv")
ret, frame = cap.read()

print(frame.shape)
# RGB: (192, 256, 3)

for i in range(100):
    frame_count = 1

    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # Split channels
    b, g, r = cv2.split(frame)

    # Check if R, G, B are equal
    if not (np.array_equal(r, g) and np.array_equal(g, b)):
        print(f"Frame {frame_count} is NOT purely grayscale")
        break
    else:
        print("All pixels are grayscale (R=G=B)")

cap.release()
