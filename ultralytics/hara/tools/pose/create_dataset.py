import os
import shutil
import random

# Define paths
source_images_dir = r'I:\chicken-pose\chicken_data\images'
source_labels_dir = 'converted-yolo-labels'
dest_base_dir = 'chicken-pose0'

# Create directories
train_images_dir = os.path.join(dest_base_dir, 'images/train')
train_labels_dir = os.path.join(dest_base_dir, 'labels/train')
val_images_dir = os.path.join(dest_base_dir, 'images/val')
val_labels_dir = os.path.join(dest_base_dir, 'labels/val')

for dir in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
    os.makedirs(dir, exist_ok=True)

# Get all image filenames
images = [f for f in os.listdir(source_images_dir) if f.endswith('.png')]
random.shuffle(images)

# Split dataset
split_idx = int(0.8 * len(images))
train_images = images[:split_idx]
val_images = images[split_idx:]

# Function to copy files
def copy_files(files, src_dir, dest_dir):
    for f in files:
        shutil.copy(os.path.join(src_dir, f), dest_dir)

# Copy training images and labels
copy_files(train_images, source_images_dir, train_images_dir)
copy_files([f.replace('.png', '.txt') for f in train_images], source_labels_dir, train_labels_dir)

# Copy validation images and labels
copy_files(val_images, source_images_dir, val_images_dir)
copy_files([f.replace('.png', '.txt') for f in val_images], source_labels_dir, val_labels_dir)

print("Dataset split and copied successfully.")
