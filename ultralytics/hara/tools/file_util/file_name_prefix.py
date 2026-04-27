import os

folder_path = r"D:\chicken_project\experiment5reid\obb_training\20250826T000000Z_20250826T002000Z_RGB_mock_10min"
prefix = "20250826T000000Z_20250826T002000Z_RGB_mock_"

count = 0

for filename in os.listdir(folder_path):
    old_path = os.path.join(folder_path, filename)

    # Skip directories
    if not os.path.isfile(old_path):
        continue

    new_filename = prefix + filename
    new_path = os.path.join(folder_path, new_filename)

    os.rename(old_path, new_path)
    count += 1

print(f"Renamed {count} files.")