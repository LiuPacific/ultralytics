import cv2
from objdetector import Detector
import numpy as np

VIDEO_PATH = r'D:\chicken_project\experiment3obb\prediction\mock_20250825T232000Z_20250825T234000Z_prediction.mkv'
RESULT_PATH = 'result.mp4'

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Detections:
    def __init__(self):
        self.detections = []

    def add(self, xyxyxyxy, confidence, class_id, tracker_id):
        self.detections.append((xyxyxyxy, confidence, class_id, tracker_id))

def draw_trail(output_image_frame, trail_points, trail_color, trail_length=50):
    for i in range(len(trail_points)):
        if len(trail_points[i]) > 1:
            for j in range(1, len(trail_points[i])):
                cv2.line(output_image_frame, (int(trail_points[i][j-1][0]), int(trail_points[i][j-1][1])),
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

    detector = Detector()
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

        # detections = Detections()
        # output_image_frame, list_bboxs = objtracker.update(detector, im)
        detections = detector.detect(im)
        # for item_bbox in detected_bboxes:
        #     x1, y1, x2, y2, x3, y3, x4,y4, conf, track_id = item_bbox
        #     detections.add((x1, y1, x2, y2), None, None, track_id)

        for detection in detections:
            poly = np.array(detection[0], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(im, [poly], True, (0, 255, 0), 2)

        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc(
                'm', 'p', '4', 'v')  # opencv3.0
            videoWriter = cv2.VideoWriter(
                RESULT_PATH, fourcc, fps, (im.shape[1], im.shape[0]))

        videoWriter.write(im)

        height, width = im.shape[:2]
        im = cv2.resize(im, (int(width/3), int(height/3)))
        cv2.imshow('Demo', im)
        cv2.waitKey(1)

    capture.release()
    videoWriter.release()
    cv2.destroyAllWindows()



