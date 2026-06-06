from pathlib import Path
import json
import re

folder = Path(r"D:\chicken_project\experiment5reid\reid_training\20250826T000000Z_20250826T002000Z_RGB_mock")  # change this



start_name = "frame_00_07_21.json"
end_name = "frame_00_20_01.json"

for json_file in sorted(folder.glob("frame_*.json")):
    if start_name <= json_file.name <= end_name:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        changed = False

        for shape in data.get("shapes", []):
            if shape.get("label") == "14":
                shape["label"] = "11"
                changed = True
            elif shape.get("label") == "11":
                shape["label"] = "14"
                changed = True

        if changed:
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            print(f"Updated: {json_file.name}")

