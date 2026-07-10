import json
from pathlib import Path

# Change this to your folder path
folder = Path(r"D:\chicken_project\experiment6report\10-min-hbb-id0\08_mock_1-15")

for json_file in folder.glob("*.json"):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # X-anylabeling usually stores labels inside "shapes"
    if "shapes" in data:
        for shape in data["shapes"]:
            shape["label"] = "0"

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Updated: {json_file.name}")

print("Done.")