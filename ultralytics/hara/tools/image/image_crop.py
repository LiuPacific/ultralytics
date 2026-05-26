import json
import cv2
import numpy as np
from pathlib import Path


def crop_obb_with_x_anylabeling_points(image, points):
    """
    Crop one rotated rectangle using the original X-AnyLabeling point order.
    """

    pts = np.array(points, dtype="float32")

    p0, p1, p2, p3 = pts

    width_1 = np.linalg.norm(p1 - p0)
    width_2 = np.linalg.norm(p2 - p3)
    crop_width = int(round(max(width_1, width_2)))

    height_1 = np.linalg.norm(p2 - p1)
    height_2 = np.linalg.norm(p3 - p0)
    crop_height = int(round(max(height_1, height_2)))

    if crop_width <= 0 or crop_height <= 0:
        raise ValueError("Invalid OBB size.")

    dst = np.array([
        [0, 0],
        [crop_width - 1, 0],
        [crop_width - 1, crop_height - 1],
        [0, crop_height - 1],
    ], dtype="float32")
    # dst = np.array([
    #     [0, crop_height - 1],              # p0
    #     [0, 0],                            # p1
    #     [crop_width - 1, 0],               # p2
    #     [crop_width - 1, crop_height - 1]  # p3
    # ], dtype="float32")

    M = cv2.getPerspectiveTransform(pts, dst)

    cropped = cv2.warpPerspective(
        image,
        M,
        (crop_width, crop_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return cropped


def crop_chickens_from_one_json(json_path, image_dir, output_dir):
    json_path = Path(json_path)
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_name = data["imagePath"]
    image_path = image_dir / image_name

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    image_stem = Path(image_name).stem

    count = 0

    for i, shape in enumerate(data["shapes"]):
        points = shape["points"]
        label = shape.get("label", str(i))

        if len(points) != 4:
            print(f"Skip shape {i}: not 4 points.")
            continue

        cropped = crop_obb_with_x_anylabeling_points(image, points)

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
    # json_dir = r"D:\chicken_project\experiment5reid\image_crop_experiment"
    # image_dir = r"D:\chicken_project\experiment5reid\image_crop_experiment"
    # output_dir = r"D:\chicken_project\experiment5reid\image_crop_experiment\out"
    json_dir = r"D:\chicken_project\experiment5reid\reid_training\combination_id"
    image_dir = r"D:\chicken_project\experiment5reid\reid_training\combination_id"
    output_dir = r"D:\chicken_project\experiment5reid\reid_training\combination_id_croped"

    batch_crop_chickens(json_dir, image_dir, output_dir)


