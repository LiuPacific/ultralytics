import subprocess
from pathlib import Path
from PIL import Image, ImageDraw
import os

"""
function:

- extracts frames from both .mkv videos at
    - 00:01
    - 01:01
    - 02:01
    ...
- saves the individual frames
- also creates a side-by-side comparison image for each timestamp
"""

"""
how to use:

1. Install FFmpeg and make sure ffmpeg and ffprobe are in your system PATH.

2. Replace:
video1_path = r"video1.mkv"
video2_path = r"video2.mkv"

3. Run the script.

Output

It will create a folder like this:

captured_frames/
    video1/
        frame_00_00_01.png
        frame_00_01_01.png
        frame_00_02_01.png
        ...
    video2/
        frame_00_00_01.png
        frame_00_01_01.png
        frame_00_02_01.png
        ...
    comparison/
        compare_00_00_01.png
        compare_00_01_01.png
        compare_00_02_01.png
        ...
"""

import subprocess
from pathlib import Path
from PIL import Image, ImageDraw


def get_video_duration(video_path):
    """
    Get video duration in seconds using ffprobe.
    """
    cmd = [
        "ffprobe.exe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video_path}:\n{result.stderr}")
    return float(result.stdout.strip())


def extract_frame(video_path, time_sec, output_path):
    """
    Extract one frame at a specific time using ffmpeg.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(time_sec),
        "-i", str(video_path),
        "-frames:v", "1",
        str(output_path)
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {video_path} at {time_sec}s:\n{result.stderr}")


def seconds_to_label(seconds):
    """
    Convert seconds to HH_MM_SS style label.
    Example: 61 -> 00_01_01
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}_{m:02d}_{s:02d}"


def seconds_to_timestamp(seconds):
    """
    Convert seconds to HH:MM:SS for display.
    Example: 61 -> 00:01:01
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def make_side_by_side(image1_path, image2_path, output_path, title_left="Video 1", title_right="Video 2", timestamp=""):
    """
    Create a side-by-side comparison image from two extracted frames.
    """
    img1 = Image.open(image1_path).convert("RGB")
    img2 = Image.open(image2_path).convert("RGB")

    # Resize to same height while keeping aspect ratio
    target_height = max(img1.height, img2.height)

    def resize_to_height(img, height):
        new_width = int(img.width * height / img.height)
        return img.resize((new_width, height), Image.LANCZOS)

    img1 = resize_to_height(img1, target_height)
    img2 = resize_to_height(img2, target_height)

    padding = 20
    top_bar = 50
    gap = 20

    canvas_width = img1.width + img2.width + gap + padding * 2
    canvas_height = target_height + top_bar + padding * 2

    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)

    # Paste images
    x1 = padding
    y1 = top_bar + padding
    x2 = padding + img1.width + gap
    y2 = top_bar + padding

    canvas.paste(img1, (x1, y1))
    canvas.paste(img2, (x2, y2))

    # Simple labels
    draw.text((x1, 15), f"{title_left} | {timestamp}", fill="black")
    draw.text((x2, 15), f"{title_right} | {timestamp}", fill="black")

    canvas.save(output_path)


# video1_type: RGB_sick
def extract_same_time_frames_with_comparison(video1, video1_type, video2, video2_type, chicken_group="",
                                             output_dir="output_frames"):
    video1 = Path(video1)
    video2 = Path(video2)
    output_dir = Path(output_dir)

    video1_dir = output_dir / (video1.stem + "_" + video1_type)
    video2_dir = output_dir / (video2.stem + "_" + video2_type)
    compare_dir = output_dir / (video1.stem + "_comparison_" + chicken_group)

    video1_dir.mkdir(parents=True, exist_ok=True)
    video2_dir.mkdir(parents=True, exist_ok=True)
    compare_dir.mkdir(parents=True, exist_ok=True)

    duration1 = get_video_duration(video1)
    duration2 = get_video_duration(video2)
    max_time = min(duration1, duration2)

    k = 0
    while True:
        time_sec = 10 * k + 1  # 00:01, 00:11, 00:21, ...
        if time_sec > max_time:
            break

        label = seconds_to_label(time_sec)
        timestamp = seconds_to_timestamp(time_sec)

        frame1_path = video1_dir / f"frame_{label}.png"
        frame2_path = video2_dir / f"frame_{label}.png"
        comparison_path = compare_dir / f"compare_{label}.png"

        print(f"Extracting frames at {timestamp} ...")

        extract_frame(video1, time_sec, frame1_path)
        extract_frame(video2, time_sec, frame2_path)

        make_side_by_side(
            frame1_path,
            frame2_path,
            comparison_path,
            title_left=video1.stem,
            title_right=video2.stem,
            timestamp=timestamp
        )

        k += 1

    print("Done.")


def experiment1():
    video1_path = r"D:\chicken_project\experiment2\RGB_sick\20250825T102000Z_20250825T104000Z.mkv"
    video2_path = r"D:\chicken_project\experiment2\Thermal sick\20250825T102000Z_20250825T104000Z.mkv"

    extract_same_time_frames_with_comparison(
        video1_path,
        "RGB_sick",
        video2_path,
        "Thermal_sick",
        output_dir=".captured_frames"
    )


def experiment2mock():
    dir_path = r"D:\chicken_project\experiment2\RGB_mock"
    file_names = os.listdir(dir_path)

    for file_name in file_names:
        video1_type = "RGB_mock"
        video1_path = os.path.join(r"D:\chicken_project\experiment2", video1_type, file_name)
        video2_type = "Thermal_mock"
        video2_path = os.path.join(r"D:\chicken_project\experiment2", video2_type, file_name)
        chicken_group = "mock"
        extract_same_time_frames_with_comparison(
            video1_path,
            video1_type,
            video2_path,
            video2_type,
            chicken_group,
            output_dir=".captured_frames"
        )


def experiment2sick():
    dir_path = r"D:\chicken_project\experiment2\RGB_sick"
    file_names = os.listdir(dir_path)

    for file_name in file_names:
        video1_type = "RGB_sick"
        video1_path = os.path.join(r"D:\chicken_project\experiment2", video1_type, file_name)
        video2_type = "Thermal_sick"
        video2_path = os.path.join(r"D:\chicken_project\experiment2", video2_type, file_name)
        chicken_group = "sick"
        extract_same_time_frames_with_comparison(
            video1_path,
            video1_type,
            video2_path,
            video2_type,
            chicken_group,
            output_dir=".captured_frames"
        )


if __name__ == "__main__":
    # experiment1()
    experiment2mock()
    experiment2sick()
