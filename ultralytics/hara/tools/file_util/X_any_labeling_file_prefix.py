from pathlib import Path
import json
import os

# Change this to your folder path
folder = Path(r"D:\chicken_project\experiment5reid\hbb_training\combination_id\12_RGB_sick")

prefix = "12_RGB_sick"

# Step 1: Update imagePath inside JSON files
for json_file in folder.glob("*.json"):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "imagePath" in data:
        image_name = data["imagePath"]

        # Only add prefix if it does not already have it
        if not image_name.startswith(prefix):
            data["imagePath"] = prefix + image_name

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# Step 2: Rename all JSON and PNG files
for file in folder.iterdir():
    if file.suffix.lower() in [".json", ".png"]:
        if not file.name.startswith(prefix):
            new_name = prefix + file.name
            new_path = file.with_name(new_name)

            if new_path.exists():
                print(f"Skipped because target already exists: {new_path.name}")
            else:
                os.rename(file, new_path)
                print(f"Renamed: {file.name} -> {new_name}")


