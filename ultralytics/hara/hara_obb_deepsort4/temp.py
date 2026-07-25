import os

folder_path = r"D:\chicken_project\experiment5reid\reid_hbb_training\20251007T000000Z_20251007T010000Z_RGB_mock 76-90\keep"

import os
import json

prefix = "10_RGB_mock"

for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        old_path = os.path.join(folder_path, filename)

        # Read JSON content
        with open(old_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Update imagePath
        if "imagePath" in data and data["imagePath"].startswith(prefix):
            old_image_path = data["imagePath"]
            data["imagePath"] = data["imagePath"][len(prefix):]

            print(f'imagePath: {old_image_path} -> {data["imagePath"]}')

        # Save updated JSON
        with open(old_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Rename JSON file by removing prefix
        if filename.startswith(prefix):
            new_filename = filename[len(prefix):]
            new_path = os.path.join(folder_path, new_filename)

            os.rename(old_path, new_path)

            print(f"File: {filename} -> {new_filename}")