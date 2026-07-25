import json
import argparse
from pathlib import Path
from copy import deepcopy


def get_json_file_list(input_folder, recursive=False):
    """
    Get all JSON files from the input folder.
    """
    input_folder = Path(input_folder)

    if recursive:
        json_files = sorted(input_folder.rglob("*.json"))
    else:
        json_files = sorted(input_folder.glob("*.json"))

    return json_files


def convert_json_file(input_json_path, input_folder, output_folder):
    """
    Convert one X-AnyLabeling JSON file:
    - shape_type == "rotation" will be converted to "rectangle"
    - existing rectangle labels will be kept unchanged
    """
    input_json_path = Path(input_json_path)
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_width = data.get("imageWidth", None)
    image_height = data.get("imageHeight", None)

    new_data = deepcopy(data)
    new_shapes = []

    converted_count = 0
    kept_count = 0

    for shape in data.get("shapes", []):
        new_shape = deepcopy(shape)

        if shape.get("shape_type") == "rotation":
            points = shape.get("points", [])

            if len(points) != 4:
                print(f"[WARNING] {input_json_path.name}: label {shape.get('label')} does not have 4 points. Skipped.")
                new_shapes.append(new_shape)
                kept_count += 1
                continue

            xs = [p[0] for p in points]
            ys = [p[1] for p in points]

            xmin = min(xs)
            xmax = max(xs)
            ymin = min(ys)
            ymax = max(ys)

            # Clamp to image boundary
            if image_width is not None:
                xmin = max(0, min(xmin, image_width - 1))
                xmax = max(0, min(xmax, image_width - 1))

            if image_height is not None:
                ymin = max(0, min(ymin, image_height - 1))
                ymax = max(0, min(ymax, image_height - 1))

            hbb_points = [
                [xmin, ymin],
                [xmax, ymin],
                [xmax, ymax],
                [xmin, ymax],
            ]

            new_shape["points"] = hbb_points
            new_shape["shape_type"] = "rectangle"

            # Remove OBB-specific field
            if "direction" in new_shape:
                del new_shape["direction"]

            converted_count += 1

        else:
            kept_count += 1

        new_shapes.append(new_shape)

    new_data["shapes"] = new_shapes

    # Keep the same relative folder structure
    relative_path = input_json_path.relative_to(input_folder)
    output_json_path = output_folder / relative_path
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)

    return output_json_path, converted_count, kept_count


# python convert_folder_obb_to_hbb.py --input_folder "D:\chicken_project\experiment5reid\reid\obb_json" --output_folder "D:\chicken_project\experiment5reid\reid\hbb_json"
# python convert_folder_obb_to_hbb.py --input_folder "D:\chicken_project\experiment5reid\reid\obb_json" --output_folder "D:\chicken_project\experiment5reid\reid\hbb_json" --recursive
def main():
    parser = argparse.ArgumentParser(
        description="Convert X-AnyLabeling OBB rotation labels to HBB rectangle labels."
    )

    parser.add_argument(
        "--input_folder",
        required=False,
        help="Folder containing X-AnyLabeling JSON files.",
        default=r"D:\chicken_project\experiment5reid\reid_training_hbb\20251217T000000Z_20251217T003000Z_RGB_sick 61-75"
    )

    parser.add_argument(
        "--output_folder",
        required=False,
        help="Folder to save converted JSON files.",
        default=r"D:\chicken_project\experiment5reid\reid_training_hbb\20251217T000000Z_20251217T003000Z_RGB_sick 61-75"
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search JSON files recursively in subfolders.",
    )

    args = parser.parse_args()

    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)

    # Step 1: get JSON file list
    json_file_list = get_json_file_list(
        input_folder=input_folder,
        recursive=args.recursive
    )

    if len(json_file_list) == 0:
        print(f"No JSON files found in: {input_folder}")
        return

    total_files = 0
    total_converted = 0
    total_kept = 0

    # Step 2: loop over JSON files and convert one by one
    for json_file in json_file_list:
        output_json_path, converted_count, kept_count = convert_json_file(
            input_json_path=json_file,
            input_folder=input_folder,
            output_folder=output_folder
        )

        total_files += 1
        total_converted += converted_count
        total_kept += kept_count

        print(
            f"[OK] {json_file.name} -> {output_json_path.name} | "
            f"converted OBB: {converted_count}, kept other shapes: {kept_count}"
        )

    print("\nFinished.")
    print(f"Total JSON files: {total_files}")
    print(f"Total OBB converted to HBB: {total_converted}")
    print(f"Total other shapes kept: {total_kept}")
    print(f"Output folder: {output_folder}")


if __name__ == "__main__":
    main()