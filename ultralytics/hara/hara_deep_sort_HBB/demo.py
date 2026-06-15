import numpy as np
import objtracker
from objdetector import Detector
import cv2

def main():
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
        # detections = OBBDetections()

        output_image_frame, tracks2draw = objtracker.update(detector, im)

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



if __name__ == '__main__':

    # VIDEO_PATH = r'F:\20250903\recordings_RGB_mock\20250826T000000Z_20250826T002000Z.mkv'
    # RESULT_PATH = 'mock_20250826T000000Z_20250826T002000Z_tracking_reid.mp4'
    # main()
    #
    # VIDEO_PATH = r'F:\20250903\recordings_RGB_sick\20250826T000000Z_20250826T002000Z.mkv'
    # RESULT_PATH = 'sick_20250826T000000Z_20250826T002000Z_tracking_reid.mp4'
    # main()
    #
    #
    # # VIDEO_PATH = r'F:\20251002\RGB_mock\20251006T230000Z_20251007T000000Z.mkv'
    # VIDEO_PATH = r'F:\20251002\RGB_mock\20251007T000000Z_20251007T010000Z.mkv'
    # RESULT_PATH = 'mock_20251007T000000Z_20251007T010000Z_tracking.mp4'
    # main()
    #
    # VIDEO_PATH = r'F:\20251002\RGB_sick\20251007T000000Z_20251007T010000Z.mkv'
    # RESULT_PATH = 'sick_20251007T000000Z_20251007T010000Z_tracking_reid.mp4'
    # main()
    #
    # VIDEO_PATH = r'F:\20251002\RGB_mock\20251006T230000Z_20251007T000000Z.mkv'
    # RESULT_PATH = 'mock_20251006T230000Z_20251007T000000Z_tracking.mp4'
    # main()
    #
    #
    #
    #
    # VIDEO_PATH = r'F:\20251130\RGB_sick\20251217T000000Z_20251217T003000Z.mkv'
    # RESULT_PATH = 'sick_20251217T000000Z_20251217T003000Z_tracking_reid.mp4'
    # main()
    #
    # VIDEO_PATH = r'F:\20251130\RGB_mock\20251217T000000Z_20251217T003000Z.mkv'
    # RESULT_PATH = 'mock_20251217T000000Z_20251217T003000Z_tracking_reid.mp4'
    # main()

    # VIDEO_PATH = r'D:\chicken_project\experiment5reid\1208040829_jump.mp4'
    # RESULT_PATH = '1208040829_jump_reoccur.mp4'
    # main()


    VIDEO_PATH = r'F:\20251002\RGB_mock\seg1.mp4'
    RESULT_PATH = 'seg1_2_tracking.mp4'
    main()























