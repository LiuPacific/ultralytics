import cv2
from obj_obb_detector import ObbDetector
import numpy as np
import objtracker

# VIDEO_PATH = r'D:\chicken_project\experiment2\RGB_mock\20250825T102000Z_20250825T104000Z.mkv'
VIDEO_PATH = r'D:\chicken_project\experiment5reid\mock_20250826T000000Z_20250826T002000Z.mkv'
RESULT_PATH = 'mock_20250826T000000Z_20250826T002000Z_track630.mp4'


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class OBBDetections:
    def __init__(self):
        self.detections = []

    def add(self, xyxyxyxy, confidence, class_id, tracker_id):
        self.detections.append((xyxyxyxy, confidence, class_id, tracker_id))


def draw_trail(output_image_frame, trail_points, trail_color, trail_length=630):
    for i in range(len(trail_points)):
        if len(trail_points[i]) > 1:
            for j in range(1, len(trail_points[i])):
                cv2.line(output_image_frame, (int(trail_points[i][j - 1][0]), int(trail_points[i][j - 1][1])),
                         (int(trail_points[i][j][0]), int(trail_points[i][j][1])), trail_color[i], thickness=3)
        if len(trail_points[i]) > trail_length:
            trail_points[i].pop(0)  # Remove the oldest point from the trail


if __name__ == '__main__':
    # Initialize video capture to get video properties
    capture = cv2.VideoCapture(VIDEO_PATH)
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {VIDEO_PATH}")

    # Get video properties (width and height)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Close the video capture
    capture.release()

    detector = ObbDetector()
    capture = cv2.VideoCapture(VIDEO_PATH)
    videoWriter = None
    fps = int(capture.get(5))
    print('fps:', fps)

    # Dictionary to store the trail points of each object
    object_trails = {}

    while True:
        _, im = capture.read()
        if im is None:
            break
        detections = OBBDetections()

        output_image_frame, list_bboxs = objtracker.update(detector, im)


        for item_bbox in list_bboxs:
            xyxyxyxy,_, track_id = item_bbox
            detections.add(xyxyxyxy, None, None, track_id)

        # Add the current object's position to the trail
        for xyxyxyxy, _, _, track_id in detections.detections:
            x1=xyxyxyxy[0][0]
            y1=xyxyxyxy[0][1]
            x3=xyxyxyxy[2][0]
            y3=xyxyxyxy[2][1]
            center = Point(x=(x1+x3)/2, y=(y1+y3)/2)
            if track_id in object_trails:
                object_trails[track_id].append((center.x, center.y))
            else:
                object_trails[track_id] = [(center.x, center.y)]

        # Draw the trail for each object
        trail_colors = [(255, 0, 0)] * len(object_trails)  # Red color for all trails
        draw_trail(output_image_frame, list(object_trails.values()), trail_colors)

        # Remove trails of objects that are not detected in the current frame
        for tracker_id in list(object_trails.keys()):
            if tracker_id not in [item[3] for item in detections.detections]:
                object_trails.pop(tracker_id)

        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc(
                'm', 'p', '4', 'v')  # opencv3.0
            videoWriter = cv2.VideoWriter(
                RESULT_PATH, fourcc, fps, (output_image_frame.shape[1], output_image_frame.shape[0]))

        videoWriter.write(output_image_frame)

        height, width = output_image_frame.shape[:2]
        output_image_frame = cv2.resize(output_image_frame, (int(width/3), int(height/3)))
        cv2.imshow('Demo', output_image_frame)
        cv2.waitKey(1)


    capture.release()
    videoWriter.release()
    cv2.destroyAllWindows()


