import numpy as np
import cv2
import math

def xyxyxyxy_to_xywhr(obb_points):
    """
    Convert OBB corner points (xyxyxyxy format) to center, width, height, rotation angle format (xywhr)

    Args:
        obb_points: numpy array of shape (4, 2) representing 4 corner points [x1,y1, x2,y2, x3,y3, x4,y4]

    Returns:
        tuple: (center_x, center_y, width, height, radian)
    """
    if isinstance(obb_points, list):
        obb_points = np.array(obb_points).reshape(4, 2)

    # Calculate center point
    center_x = np.mean(obb_points[:, 0])
    center_y = np.mean(obb_points[:, 1])

    # Calculate width and height
    # Width is distance between first and second points (or third and fourth)
    width = np.linalg.norm(obb_points[0] - obb_points[1])
    height = np.linalg.norm(obb_points[1] - obb_points[2])

    # Calculate rotation angle
    # Vector from point 0 to point 1 represents the width direction
    dx = obb_points[1][0] - obb_points[0][0]
    dy = obb_points[1][1] - obb_points[0][1]
    radian = math.atan2(dy, dx)

    # No need to normalize angle since atan2 returns -pi to pi
    return center_x, center_y, width, height, radian

# def xywhr_to_xyxyxyxy(xywhr):
#     return xywhr_to_xyxyxyxy(xywhr[0],xywhr[1],xywhr[2],xywhr[3],xywhr[4])

def xywhr_to_xyxyxyxy(center_x, center_y, width, height, angle):
    """
    Convert center, width, height, rotation angle format to OBB corner points

    Args:
        center_x, center_y: center coordinates
        width, height: dimensions
        angle: rotation angle in radians

    Returns:
        numpy array: shape (4, 2) representing 4 corner points
    """
    # Angle is already in radians
    angle_rad = angle

    # Calculate half dimensions
    half_width = width / 2
    half_height = height / 2

    # Calculate corner points relative to center
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    # Corner points in order: top-left, top-right, bottom-right, bottom-left
    corners = np.array([
        [-half_width, -half_height],  # top-left
        [half_width, -half_height],   # top-right
        [half_width, half_height],    # bottom-right
        [-half_width, half_height]    # bottom-left
    ])

    # Rotate and translate corners
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated_corners = np.dot(corners, rotation_matrix.T)

    # Translate to center position
    final_corners = rotated_corners + np.array([center_x, center_y])

    return final_corners.astype(np.float32)

def obb_to_xyxy_aligned(obb_points):
    """
    Convert OBB to axis-aligned bounding box (xyxy format) for fallback matching

    Args:
        obb_points: numpy array of shape (4, 2) representing 4 corner points

    Returns:
        tuple: (x1, y1, x2, y2) axis-aligned bounding box
    """
    if isinstance(obb_points, list):
        obb_points = np.array(obb_points).reshape(4, 2)

    x_coords = obb_points[:, 0]
    y_coords = obb_points[:, 1]

    x1 = np.min(x_coords)
    y1 = np.min(y_coords)
    x2 = np.max(x_coords)
    y2 = np.max(y_coords)

    return x1, y1, x2, y2

def calculate_obb_centroid(obb_points):
    """
    Calculate centroid of OBB directly from corner points

    Args:
        obb_points: numpy array of shape (4, 2) representing 4 corner points

    Returns:
        tuple: (center_x, center_y)
    """
    if isinstance(obb_points, list):
        obb_points = np.array(obb_points).reshape(4, 2)

    center_x = np.mean(obb_points[:, 0])
    center_y = np.mean(obb_points[:, 1])

    return center_x, center_y

def rotated_iou(box1, box2):
    """
    Calculate Intersection over Union for two rotated rectangles

    Args:
        box1, box2: tuples of (center_x, center_y, width, height, angle) or numpy arrays of shape (4, 2)

    Returns:
        float: IoU value between 0 and 1
    """
    # Convert to OpenCV rotated rectangle format if needed
    if len(box1) == 5:  # xywhr format
        rect1 = ((box1[0], box1[1]), (box1[2], box1[3]), math.degrees(box1[4]))
    else:  # xyxyxyxy format
        center_x, center_y, w, h, radian = xyxyxyxy_to_xywhr(box1)
        rect1 = ((center_x, center_y), (w, h), math.degrees(radian))

    if len(box2) == 5:  # xywhr format
        rect2 = ((box2[0], box2[1]), (box2[2], box2[3]), math.degrees(box2[4]))
    else:  # xyxyxyxy format
        center_x, center_y, w, h, radian = xyxyxyxy_to_xywhr(box2)
        rect2 = ((center_x, center_y), (w, h), math.degrees(radian))

    # Use OpenCV to calculate intersection
    try:
        intersection_type, intersection_points = cv2.rotatedRectangleIntersection(rect1, rect2)

        if intersection_type == cv2.INTERSECT_NONE:
            return 0.0

        # Calculate areas
        area1 = rect1[1][0] * rect1[1][1]
        area2 = rect2[1][0] * rect2[1][1]

        if intersection_type == cv2.INTERSECT_FULL:
            intersection_area = min(area1, area2)
        else:
            # Calculate intersection polygon area
            if len(intersection_points) >= 3:
                intersection_area = cv2.contourArea(intersection_points.astype(np.float32))
            else:
                return 0.0

        union_area = area1 + area2 - intersection_area

        return intersection_area / union_area if union_area > 0 else 0.0

    except Exception as e:
        # Fallback to axis-aligned IoU if rotated calculation fails
        print(f"Warning: Rotated IoU calculation failed, using axis-aligned fallback: {e}")
        x1_1, y1_1, x2_1, y2_1 = obb_to_xyxy_aligned(box1)
        x1_2, y1_2, x2_2, y2_2 = obb_to_xyxy_aligned(box2)

        # Calculate axis-aligned IoU
        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

def extract_obb_crop(image, obb_points, padding=0):
    """
    Extract rotated crop from image using OBB points

    Args:
        image: numpy array of shape (H, W, C)
        obb_points: numpy array of shape (4, 2) representing 4 corner points
        padding: padding around the OBB

    Returns:
        numpy array: cropped and rotated image region
    """
    if isinstance(obb_points, list):
        obb_points = np.array(obb_points).reshape(4, 2)

    # Get bounding box of OBB
    x_coords = obb_points[:, 0]
    y_coords = obb_points[:, 1]
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)

    # Add padding
    x_min = max(0, int(x_min - padding))
    y_min = max(0, int(y_min - padding))
    x_max = min(image.shape[1], int(x_max + padding))
    y_max = min(image.shape[0], int(y_max + padding))

    # Extract axis-aligned crop containing the OBB
    crop = image[y_min:y_max, x_min:x_max]

    # Calculate transformation to align OBB
    center_x, center_y, width, height, radian = xyxyxyxy_to_xywhr(obb_points)

    # Adjust center coordinates relative to crop
    center_x_crop = center_x - x_min
    center_y_crop = center_y - y_min

    # Create rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((center_x_crop, center_y_crop), math.degrees(radian), 1.0)

    # Apply rotation
    rotated_crop = cv2.warpAffine(crop, rotation_matrix, (crop.shape[1], crop.shape[0]))

    # Extract the aligned rectangular region
    half_width = int(width / 2 + padding)
    half_height = int(height / 2 + padding)

    x1 = max(0, int(center_x_crop - half_width))
    y1 = max(0, int(center_y_crop - half_height))
    x2 = min(rotated_crop.shape[1], int(center_x_crop + half_width))
    y2 = min(rotated_crop.shape[0], int(center_y_crop + half_height))

    final_crop = rotated_crop[y1:y2, x1:x2]

    return final_crop

def test_rotated_iou():
    """Comprehensive test function for rotated IoU calculation."""
    print("Testing rotated_iou function...")

    # Test case 1: Identical boxes (should be 1.0)
    box1 = (100, 100, 50, 20, 0)  # cx, cy, w, h, angle in radians
    box2 = (100, 100, 50, 20, 0)
    iou = rotated_iou(box1, box2)
    print(f"Identical boxes IoU: {iou} (expected: 1.0)")
    assert abs(iou - 1.0) < 1e-6, f"Expected 1.0, got {iou}"

    # Test case 2: Slightly offset boxes (partial overlap)
    box1 = (100, 100, 50, 20, 0)
    box2 = (120, 100, 50, 20, 0)  # 20 pixel overlap
    iou = rotated_iou(box1, box2)
    print(f"Offset boxes IoU: {iou} (expected: ~0.43)")
    assert 0.4 < iou < 0.45, f"Expected ~0.43, got {iou}"

    # Test case 3: Rotated boxes (45 degrees)
    box1 = (100, 100, 50, 20, 0)
    box2 = (100, 100, 50, 20, math.radians(45))  # 45 degrees in radians
    iou = rotated_iou(box1, box2)
    print(f"45° rotated boxes IoU: {iou} (expected: > 0)")
    assert iou > 0, f"Expected > 0, got {iou}"

    # Test case 4: No overlap
    box1 = (100, 100, 50, 20, 0)
    box2 = (200, 200, 50, 20, 0)
    iou = rotated_iou(box1, box2)
    print(f"No overlap IoU: {iou} (expected: 0.0)")
    assert abs(iou - 0.0) < 1e-6, f"Expected 0.0, got {iou}"

    # Test case 5: Different sizes
    box1 = (100, 100, 40, 30, 0)
    box2 = (100, 100, 60, 20, 0)
    iou = rotated_iou(box1, box2)
    print(f"Different sizes IoU: {iou} (expected: > 0)")
    assert iou > 0, f"Expected > 0, got {iou}"

    # Test case 6: Test with xyxyxyxy format
    # Create OBB corner points for a box at (100,100) with w=50, h=20, angle=0
    corners1 = xywhr_to_xyxyxyxy(100, 100, 50, 20, 0)
    corners2 = xywhr_to_xyxyxyxy(110, 100, 50, 20, 0)
    iou = rotated_iou(corners1, corners2)
    print(f"xyxyxyxy format IoU: {iou} (expected: ~0.67)")
    assert 0.6 < iou < 0.7, f"Expected ~0.67, got {iou}"

    # Test case 7: Edge case - very small overlap
    box1 = (100, 100, 50, 20, 0)
    box2 = (149, 100, 50, 20, 0)  # Just touching
    iou = rotated_iou(box1, box2)
    print(f"Touching boxes IoU: {iou} (expected: ~0.0)")
    assert iou < 0.1, f"Expected ~0.0, got {iou}"

    # Test case 8: 90 degree rotation
    box1 = (100, 100, 50, 20, 0)
    box2 = (100, 100, 50, 20, math.radians(90))
    iou = rotated_iou(box1, box2)
    print(f"90° rotated boxes IoU: {iou} (expected: > 0)")
    assert iou > 0, f"Expected > 0, got {iou}"

    print("All tests passed!")

if __name__ == "__main__":
    test_rotated_iou()
