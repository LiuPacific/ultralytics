import json
from pathlib import Path


def remove_negative_label_shapes(json_path: Path, overwrite: bool = True):
    """
    Remove shapes whose label is a negative number.

    Example:
        "label": "-3"  -> remove this shape
        "label": "4"   -> keep this shape
        "label": "0"   -> keep this shape
    """

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    original_shapes = data.get("shapes", [])

    new_shapes = []
    removed_count = 0

    for shape in original_shapes:
        label = shape.get("label", "")

        try:
            label_number = int(label)
        except ValueError:
            # If label is not a number, keep it
            new_shapes.append(shape)
            continue

        if label_number < 0:
            removed_count += 1
        else:
            new_shapes.append(shape)

    data["shapes"] = new_shapes

    if overwrite:
        output_path = json_path
    else:
        output_path = json_path.with_name(json_path.stem + "_cleaned.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return removed_count


def process_folder(folder_path: str, overwrite: bool = True):
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    json_files = list(folder.glob("*.json"))

    total_removed = 0

    for json_file in json_files:
        removed = remove_negative_label_shapes(json_file, overwrite=overwrite)
        total_removed += removed
        print(f"{json_file.name}: removed {removed} negative-label shapes")

    print(f"\nDone. Processed {len(json_files)} JSON files.")
    print(f"Total removed shapes: {total_removed}")


if __name__ == "__main__":
    folder_path = r"D:\chicken_project\experiment5reid\reid_training\combination_id0"

    # overwrite=True means directly modify original JSON files
    # overwrite=False means create new files like xxx_cleaned.json
    process_folder(folder_path, overwrite=True)