import json
import os

NUM_KEYPOINTS = 17
# Ensure the output directory exists
output_dir = 'converted-yolo-labels'
os.makedirs(output_dir, exist_ok=True)

# Load data from 'annotations/person_keypoints_default.json'
with open(r"I:\chicken-pose\chicken_data\hara_chicken_pose_anno_cocokeys\annotations\person_keypoints_default.json", 'r') as f:
    data = json.load(f)
    

def coco_to_yolo(bbox, img_width, img_height):
    x_center, y_center, width, height = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2, bbox[2], bbox[3])
    return [x_center / img_width, y_center / img_height, width / img_width, height / img_height]

for annotation in data['annotations']:
    img_id = annotation['image_id']
    img_data = next((img for img in data['images'] if img['id'] == img_id), None)
    if not img_data: continue

    img_width, img_height = img_data['width'], img_data['height']
    bbox = coco_to_yolo(annotation['bbox'], img_width, img_height)
    
    keypoints = annotation['keypoints']
    keypoints += [0] * (NUM_KEYPOINTS*3 - len(keypoints))  # Ensure NUM_KEYPOINTS = 17 keypoints
    # Adjust keypoints visibility and normalize coordinates
    keypoints_yolo = [kp / img_width if i % 3 == 0 else kp / img_height if i % 3 == 1 else 2.0 if kp != 0 else 0 for i, kp in enumerate(keypoints)]
    
    yolo_data = [0] + bbox + keypoints_yolo

    file_name = os.path.join(output_dir, img_data['file_name'].replace('.png', '.txt'))
    file_name = file_name.replace('.jpg', '.txt')
    with open(file_name, 'w') as f_yolo:
        f_yolo.write(' '.join(map(str, yolo_data)))

print("Conversion to YOLO format completed and saved in 'converted-yolo-labels'.")


