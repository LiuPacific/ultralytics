import cv2
import re
import torch
import easyocr
from datetime import datetime, timezone


def extract_time(frame) -> int:
    # Crop top‑right region containing the timestamp
    h, w, _ = frame.shape
    crop_h, crop_w = 60, 350  # adjust these to match your overlay size
    roi = frame[0:crop_h, w - crop_w:w]

    # Run OCR on the crop with GPU
    #    EasyOCR uses PyTorch under the hood; gpu=True will offload to CUDA
    reader = easyocr.Reader(['en'], gpu=True)
    result = reader.readtext(roi, detail=0)
    ocr_text = ''.join(result)
    print("Raw OCR text:", ocr_text)

    # 4. Pull out the timestamp via regex
    m = re.search(
        r'(?P<mon>\d{2})[^\d]*(?P<day>\d{2})[^\d]*(?P<year>\d{4})[^\d]*(?P<hour>\d{2})[^\d]*(?P<min>\d{2})[^\d]*(?P<sec>\d{2})[^\d]*',
        ocr_text)
    if m:
        mon = int(m.group("mon"))
        day = int(m.group("day"))
        year = int(m.group("year"))
        hour = int(m.group("hour"))
        min = int(m.group("min"))
        sec = int(m.group("sec"))

        dt = datetime(year, mon, day, hour, min, sec, tzinfo=timezone.utc)
        unix_ts = dt.timestamp()
        print("Extracted UTC timestamp:", unix_ts)
        return unix_ts
    else:
        raise RuntimeError("No timestamp match found in OCR output.")


def get_first_time_changing_frame(cap) -> int:
    """
    from 0
    """
    prev_unix_ts = 0
    i = 0
    while True:
        _, frame = cap.read()
        if frame is None:
            print("no frame read")
            break
        unit_ts = extract_time(frame)
        if prev_unix_ts != unit_ts and prev_unix_ts != 0:
            print(f"time change at frame {i}")
            break
        prev_unix_ts = unit_ts
        i = i + 1
    return i


def get_last_time_before_changing_frame(cap, total_frame_num, fps) -> int:
    start_frame = total_frame_num - fps - 1
    if start_frame < fps * 2:
        raise RuntimeError("video too short, at least 3 seconds are required")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    prev_unix_ts = 0
    i = start_frame
    while True:
        _, frame = cap.read()
        if frame is None:
            print("no frame read")
            break
        unit_ts = extract_time(frame)
        if prev_unix_ts != unit_ts and prev_unix_ts != 0:
            print(f"time change at frame {i}")
            break
        prev_unix_ts = unit_ts
        i = i + 1
    return i - 1


if __name__ == '__main__':
    # Verify CUDA is available
    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA is not available. Make sure you have a GPU and the correct drivers installed.")

    device = torch.device('cuda')
    # Read a frame from the video
    video_path = r'./.recordings/20250718T062401Z_20250718T062407Z.mp4'
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    fps = int(cap.get(5))
    print('fps:', fps)

    total_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('total frame num: ', total_frame_num)

    first_changing_index = get_first_time_changing_frame(cap)
    last_before_changing_index = get_last_time_before_changing_frame(cap, total_frame_num, fps)

    print("first_changing_index: " + str(first_changing_index))
    print("last_before_changing_index: " + str(last_before_changing_index))
    cap.release()
