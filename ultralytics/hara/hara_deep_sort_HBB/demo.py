import numpy as np
import objtracker
from objdetector import Detector
import cv2
from ultralytics.hara.hara_deep_sort_HBB.deep_sort.configs.common_cfg import cfg
from deep_sort.deep_sort import DeepSort

def get_deepsort():
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True, use_reid=cfg.DEEPSORT.USE_REID,
                        tracking_csv_path=cfg.DEEPSORT.TRACKING_CSV_PATH)
    return deepsort

def start():
    # Initialize video capture to get video properties
    capture = cv2.VideoCapture(cfg.DEEPSORT.VIDEO_PATH)
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {cfg.DEEPSORT.VIDEO_PATH}")

    # Get video properties (width and height)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Close the video capture
    capture.release()

    detector = Detector()
    capture = cv2.VideoCapture(cfg.DEEPSORT.VIDEO_PATH)
    videoWriter = None
    fps = int(capture.get(5))
    print('fps:', fps)

    # Dictionary to store the trail points of each object
    object_trails = {}
    deepsort_obb = get_deepsort()
    while True:
        _, im = capture.read()
        if im is None:
            break
        # detections = OBBDetections()

        output_image_frame, tracks2draw = objtracker.update(detector, im, deepsort_obb)

        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc(
                'm', 'p', '4', 'v')  # opencv3.0
            videoWriter = cv2.VideoWriter(
                cfg.DEEPSORT.RESULT_PATH, fourcc, fps, (output_image_frame.shape[1], output_image_frame.shape[0]))

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

    # VIDEO_PATH = r'F:\20251002\RGB_mock\seg1.mp4'
    # RESULT_PATH = 'seg1_4_csv.mp4'
    # cfg.merge_from_file(r"deep_sort/configs/deep_sort.yaml")
    cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\HBB5min\hold1_tracking.yaml")
    start()
    cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\HBB5min\hold2_tracking.yaml")
    start()
    cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\HBB5min\hold3_tracking.yaml")
    start()
    cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\HBB5min\hold4_tracking.yaml")
    start()
    cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\HBB5min\hold5_tracking.yaml")
    start()
    cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\HBB5min\hold6_tracking.yaml")
    start()
























