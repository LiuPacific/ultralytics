import cv2
import os

# Input and output directories
input_dir = r'G:\project_chicken\code\experiment_deepSORT\chicken_label128128\test\0001'    # Replace with your actual input path
output_dir = r'G:\project_chicken\code\experiment_deepSORT\chicken_label12864\test\0001'  # Replace with your desired output path

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop through all jpg images in the input directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith(".jpg"):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)

        if img is not None:
            resized_img = cv2.resize(img, (64, 128))  # Width x Height
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, resized_img)
        else:
            print(f"Warning: Failed to read {img_path}")
