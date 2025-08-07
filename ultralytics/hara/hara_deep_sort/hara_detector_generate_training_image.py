import numpy as np
from objdetector import Detector
import cv2
import torch
import time

# VIDEO_PATH = r'G:\project_chicken\code\experiment_deepSORT\video\test_person.mp4'
VIDEO_PATH = r'E:\chicken_project\workspace\ultralytics\ultralytics\hara\hara_deep_sort\.video\20250513_111230.mp4'


def _xywh_to_xyxy(bbox_xywh, img_height, img_width):
    x, y, w, h = bbox_xywh
    x1 = max(int(x - w / 2), 0)
    x2 = min(int(x + w / 2), img_width - 1)
    y1 = max(int(y - h / 2), 0)
    y2 = min(int(y + h / 2), img_height - 1)
    return x1, y1, x2, y2


if __name__ == '__main__':
    # Initialize video capture to get video properties
    capture = cv2.VideoCapture(VIDEO_PATH)
    if not capture.isOpened():
        print("Error opening video file.")
        exit()

    # Get video properties (width and height)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Close the video capture
    capture.release()

    detector = Detector()
    capture = cv2.VideoCapture(VIDEO_PATH)
    videoWriter = None
    fps = int(capture.get(5))
    print('fps:', fps)

    # Dictionary to store the trail points of each object
    object_trails = {}

    i = 0
    while True:
        _, im = capture.read()
        if im is None:
            break

        bbox_xywh = []
        confs = []
        bboxes2draw = []

        i = i + 1
        j = 0
        if i%30 == 0:
            _, bboxes = detector.detect(im)
            if len(bboxes):
                # Adapt detections to deep sort input format
                for x1, y1, x2, y2, _, conf in bboxes:
                    obj = [
                        int((x1 + x2) / 2), int((y1 + y2) / 2),
                        x2 - x1, y2 - y1
                    ]
                    bbox_xywh.append(obj)
                    confs.append(conf)


                    cropped_image = im[int(y1):int(y2), int(x1):int(x2)]
                    new_width, new_height = 128, 128
                    resized_img = cv2.resize(cropped_image, (new_width, new_height))

                    j = j + 1


                    output_filename = f'./.temp/resized_img{i}_{j}.jpg'
                    save_status = cv2.imwrite(output_filename, resized_img)
                    if save_status:
                        print(f"Successfully saved the cropped image as '{output_filename}'")
                        # cv2.imshow('Original Image', im)
                        # cv2.imshow('cropped resized_img Image', resized_img)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                    else:
                        print("Error: Could not save the image.")







            else:
                print('no bbox in this frame')
