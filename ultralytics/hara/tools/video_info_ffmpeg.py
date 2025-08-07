import subprocess
import json

def get_i_frame_indices(video_path):
    cmd = [
        "ffprobe",
        "-select_streams", "v",
        "-show_frames",
        "-show_entries", "frame=key_frame,coded_picture_number",
        "-of", "json",
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    frames = json.loads(result.stdout)["frames"]

    print(frames)
    i_frames = [f["coded_picture_number"] for f in frames if f["key_frame"] == 1]
    return i_frames

# Example usage
# video_path = r"../.recordings/20250724T133000Z_20250724T140000Z.mp4"
video_path = r"E:\chicken_project\workspace\Chicken\hara_video\hara_onvif\.recordings\20250726T050050Z_20250726T050100Z.mp4"
# video_path = r"E:\chicken_project\workspace\Chicken\hara_video\hara_onvif\.recordings\20250724T133000Z_20250724T140000Z.mp4"
i_frames = get_i_frame_indices(video_path)

# Print GOP lengths
gop_lengths = [j - i for i, j in zip(i_frames[:-1], i_frames[1:])]
print("I-Frame positions:", i_frames)
print("GOP lengths:", gop_lengths)

