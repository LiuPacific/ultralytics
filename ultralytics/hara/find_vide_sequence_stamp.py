import cv2
import re
import torch
import easyocr
from datetime import datetime, timezone






cap_RGB = cv2.VideoCapture("./.recordings/20250705_210606_thermal.mp4")
# cap_RGB = cv2.VideoCapture("./.recordings/20250718T192731Z_20250718T192800Z.mp4")
# cap_thermal = cv2.VideoCapture("./.recordings/20250718T192731Z_20250718T192800Z_thermal.mp4")


# Get timestamps from each frame
timestampsA = []
for i in range(100):
    ret, frame = cap_RGB.read()
    if frame is None:
        print("frame is None")
        break
    timestamp = cap_RGB.get(cv2.CAP_PROP_POS_MSEC)  # in milliseconds
    print(f"{i} : {timestamp}")
    timestampsA.append((timestamp, frame))




