import json
import csv
import re
from pathlib import Path
import argparse

"""
For one JSON file like: 08_RGB_mockframe_00_02_31.json
the script will output: time_seconds = 151
because: 00 * 3600 + 02 * 60 + 31 = 151

For a shape with: "label": "-2"
the output will be: chicken_id = 2; invisible = 1

For: "label": "2"
the output will be: chicken_id = 2; invisible = 0
"""

def parse_time_seconds_from_filename(json_path: Path) -> int:
    """
    Parse time from filename.

    Example:
        08_RGB_mockframe_00_02_31.json

    The last three numbers mean:
        hour = 00
        minute = 02
        second = 31

    Return:
        total seconds = hour * 3600 + minute * 60 + second
    """
    numbers = re.findall(r"\d+", json_path.stem)

    if len(numbers) < 3:
        raise ValueError(f"Cannot parse time from filename: {json_path.name}")

    hour, minute, second = map(int, numbers[-3:])
    return hour * 3600 + minute * 60 + second


def parse_chicken_label(label_value):
    """
    Positive label:
        "2"  -> chicken_id = 2, invisible = 0

    Negative label:
        "-2" -> chicken_id = 2, invisible = 1
    """
    label_str = str(label_value).strip()

    try:
        label_int = int(label_str)
    except ValueError:
        raise ValueError(f"Invalid label value: {label_value}")

    if label_int < 0:
        return abs(label_int), 1
    else:
        return label_int, 0


def compute_center(points):
    """
    Compute OBB center from corner points.

    points example:
        [
            [x1, y1],
            [x2, y2],
            [x3, y3],
            [x4, y4]
        ]

    For a rectangle/parallelogram OBB, the center is the average
    of the four corner coordinates.
    """
    if not points:
        return None, None

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    center_x = sum(xs) / len(xs)
    center_y = sum(ys) / len(ys)

    return center_x, center_y


def extract_tracks(input_folder: Path, output_csv: Path):
    rows = []

    json_files = sorted(input_folder.rglob("*.json"))

    for json_file in json_files:
        try:
            time_seconds = parse_time_seconds_from_filename(json_file)
        except ValueError as e:
            print(f"[SKIP] {e}")
            continue

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[SKIP] Cannot read {json_file}: {e}")
            continue

        shapes = data.get("shapes", [])

        for shape in shapes:
            label_value = shape.get("label")

            if label_value is None:
                continue

            try:
                chicken_id, invisible = parse_chicken_label(label_value)
            except ValueError as e:
                print(f"[SKIP] {json_file.name}: {e}")
                continue

            points = shape.get("points", [])
            center_x, center_y = compute_center(points)

            direction = shape.get("direction", "")

            rows.append({
                "time_seconds": time_seconds,
                "chicken_id": chicken_id,
                "x": center_x,
                "y": center_y,
                "direction": direction,
                "invisible": invisible,
                "frame": time_seconds*30
            })

    rows.sort(key=lambda r: (r["time_seconds"], r["chicken_id"]))

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "time_seconds",
                "chicken_id",
                "x",
                "y",
                "direction",
                "invisible",
                "frame"
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. Extracted {len(rows)} records.")
    print(f"Saved to: {output_csv}")


def main():
    input_folder = Path(r"D:\chicken_project\experiment6report\10-min-obb\12_sick_61-75")
    output_csv = Path(r"D:\chicken_project\experiment6report\tracking\12_sick_61-75_tracking.csv")


    extract_tracks(input_folder, output_csv)


if __name__ == "__main__":
    main()