import json
import cv2
import numpy as np
from pathlib import Path


def crop_hbb_with_points(image, points):
    """Crop the horizontal bounding box that encloses an annotation's points."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
        raise ValueError("Invalid HBB points.")

    height, width = image.shape[:2]
    x_min = max(0, int(np.floor(np.min(pts[:, 0]))))
    y_min = max(0, int(np.floor(np.min(pts[:, 1]))))
    x_max = min(width, int(np.ceil(np.max(pts[:, 0]))))
    y_max = min(height, int(np.ceil(np.max(pts[:, 1]))))

    if x_max <= x_min or y_max <= y_min:
        raise ValueError("Invalid HBB size.")

    return image[y_min:y_max, x_min:x_max].copy()


def crop_chickens_from_one_json(json_path, image_dir, output_dir):
    json_path = Path(json_path)
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_name = data.get("imagePath") or f"{json_path.stem}.png"
    image_path = Path(image_name)
    if not image_path.is_absolute():
        image_path = image_dir / image_name

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    image_stem = Path(image_name).stem

    count = 0

    for i, shape in enumerate(data["shapes"]):
        points = shape.get("points", [])
        label = shape.get("label", str(i))

        try:
            cropped = crop_hbb_with_points(image, points)
        except ValueError as exc:
            print(f"Skip shape {i}: {exc}")
            continue

        save_name = f"id_{label}_{image_stem}.png"
        save_path = output_dir / save_name

        cv2.imwrite(str(save_path), cropped)
        count += 1

    print(f"Finished {json_path.name}: saved {count} chicken crops.")


def batch_crop_chickens(json_dir, image_dir, output_dir):
    json_dir = Path(json_dir)

    json_files = sorted(json_dir.glob("*.json"))

    for json_path in json_files:
        crop_chickens_from_one_json(
            json_path=json_path,
            image_dir=image_dir,
            output_dir=output_dir
        )


if __name__ == "__main__":
    source_dir = Path(r"D:\chicken_project\experiment6report\reid\hbb\10-min-hbb\12_mock_46-60")
    json_dir = source_dir
    image_dir = source_dir
    output_dir = source_dir.with_name(f"{source_dir.name}_cropped")

    batch_crop_chickens(json_dir, image_dir, output_dir)
