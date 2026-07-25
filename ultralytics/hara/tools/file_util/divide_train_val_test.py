import os
import random
import shutil
from pathlib import Path

# =========================
# Settings
# =========================

source_dir = Path(r"D:\chicken_project\experiment5reid\reid_obb_training\combination_id_0614_half\combination_id0")

train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

random_seed = 42

# If True, copy files.
# If False, move files.
copy_files = True

image_extensions = [".png", ".jpg", ".jpeg"]

# =========================
# Create output folders
# =========================

train_dir = source_dir / "train"
val_dir = source_dir / "val"
test_dir = source_dir / "test"

for folder in [train_dir, val_dir, test_dir]:
    folder.mkdir(exist_ok=True)

# =========================
# Find complete image-txt-json groups
# =========================

samples = []

for image_path in source_dir.iterdir():
    if image_path.suffix.lower() in image_extensions:
        base_name = image_path.stem

        txt_path = source_dir / f"{base_name}.txt"
        json_path = source_dir / f"{base_name}.json"

        if txt_path.exists() and json_path.exists():
            samples.append({
                "base_name": base_name,
                "image": image_path,
                "txt": txt_path,
                "json": json_path
            })
        else:
            print(f"Warning: missing txt or json for {image_path.name}")

print(f"Total complete samples found: {len(samples)}")

# =========================
# Random shuffle
# =========================

random.seed(random_seed)
random.shuffle(samples)

# =========================
# Split samples
# =========================

total = len(samples)

train_count = int(total * train_ratio)
val_count = int(total * val_ratio)

train_samples = samples[:train_count]
val_samples = samples[train_count:train_count + val_count]
test_samples = samples[train_count + val_count:]

print(f"Train samples: {len(train_samples)}")
print(f"Val samples:   {len(val_samples)}")
print(f"Test samples:  {len(test_samples)}")

# =========================
# Copy or move files
# =========================

def transfer_sample(sample, target_dir):
    files = [sample["image"], sample["txt"], sample["json"]]

    for file_path in files:
        target_path = target_dir / file_path.name

        if copy_files:
            shutil.copy2(file_path, target_path)
        else:
            shutil.move(str(file_path), str(target_path))


for sample in train_samples:
    transfer_sample(sample, train_dir)

for sample in val_samples:
    transfer_sample(sample, val_dir)

for sample in test_samples:
    transfer_sample(sample, test_dir)

print("Done!")