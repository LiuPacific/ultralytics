from pathlib import Path
import re

# In X-any-labeling, I labeled each chicken with different ID.
# But here I just want to convert all IDs to 0, so that I can use the converted txt files for training.
# So in YOLO OBB training file, each line's first number is converted into 0 from ID.

# Change this to your folder path
INPUT_DIR = Path(r"D:\chicken_project\experiment5reid\obb_training\20250826T000000Z_20250826T002000Z_RGB_mock_10min_labels_id0")

# Keep original filenames.
# If IN_PLACE = True, original files are directly modified.
# If IN_PLACE = False, changed files are saved to OUTPUT_DIR with the same filenames.
IN_PLACE = False
OUTPUT_DIR = INPUT_DIR / "converted"

# Match an integer only at the beginning of a line, followed by whitespace
pattern = re.compile(r"^(\s*)\d+(?=\s)")


def convert_file(input_path: Path, output_path: Path) -> int:
    lines = input_path.read_text(encoding="utf-8").splitlines(keepends=True)

    new_lines = []
    changed_count = 0

    for line in lines:
        match = pattern.search(line)

        if match:
            original_integer = match.group(0).strip()

            # Count only when the original integer is not already 0
            if original_integer != "0":
                changed_count += 1

            new_line = pattern.sub(r"\g<1>0", line, count=1)
        else:
            new_line = line

        new_lines.append(new_line)

    output_path.write_text("".join(new_lines), encoding="utf-8")
    return changed_count


def main() -> None:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Folder not found: {INPUT_DIR}")

    txt_files = list(INPUT_DIR.glob("*.txt"))
    if not txt_files:
        print("No .txt files found.")
        return

    if not IN_PLACE:
        OUTPUT_DIR.mkdir(exist_ok=True)

    total_changed = 0

    for txt_file in txt_files:
        if IN_PLACE:
            output_file = txt_file
        else:
            # Keep the same original filename in the output folder
            output_file = OUTPUT_DIR / txt_file.name

        changed_count = convert_file(txt_file, output_file)
        total_changed += changed_count

        print(f"{txt_file.name}: changed {changed_count} integer(s)")

    print(f"Done. Total changed integers: {total_changed}")


if __name__ == "__main__":
    main()
