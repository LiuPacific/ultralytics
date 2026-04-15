# OBB DeepSORT Implementation Guide

## Overview

This document provides a complete guide to the OBB (Oriented Bounding Box) DeepSORT implementation integrated into the `hara_obb_deepsort` package.

## What's Been Implemented

### 1. **OBB Format Utilities** (`utils/obb_utils.py`)
Provides conversion functions between different OBB representations:
- **xyxyxyxy_to_xywhr()**: Convert 4 corner points to center + dimensions + angle
- **xywhr_to_xyxyxyxy()**: Reverse conversion
- **obb_to_xyxy_aligned()**: Get axis-aligned bounding box (fallback)
- **rotated_iou()**: Calculate rotated Intersection over Union
- **extract_obb_crop()**: Extract rotated rectangular crops from images
- **calculate_obb_centroid()**: Get center point of OBB

### 2. **Extended Kalman Filter** (`utils/kalman_filter_obb.py`)
10-dimensional Kalman filter for OBB tracking:
- **State**: `[cx, cy, w, h, angle, vcx, vcy, vw, vh, vangle]`
- Handles rotation angle with proper angle wrapping (-180 to 180 degrees)
- Methods: `initiate()`, `predict()`, `update()`, `project()`, `gating_distance()`

### 3. **OBB Detection Class** (`utils/detection_obb.py`)
Wrapper for OBB detections:
```python
detection = OBBDetection(xyxyxyxy, confidence, feature)
# Access in different formats:
detection.to_xywhr()        # Get (cx, cy, w, h, angle)
detection.to_xyxyxyxy()     # Get corner points
detection.to_xyah()         # Get (cx, cy, aspect_ratio, h) for compatibility
```

### 4. **Rotated IOU Matching** (`utils/iou_matching_obb.py`)
Two IOU cost functions for matching:
- **obb_iou_cost()**: Uses precise rotated IOU with OpenCV
- **iou_cost_fallback()**: Uses axis-aligned IOU (faster, less accurate)

### 5. **OBB-Aware Tracker** (`utils/tracker_obb.py`)
Enhanced tracker with:
- OBB-specific matching cascade
- Configurable IOU cost function selection
- Proper track initialization for OBB state space

### 6. **OBB Track Class** (`utils/track_obb.py`)
Track object with OBB state representation:
- Methods: `to_xywhr()`, `to_xyxyxyxy()`, `to_tlwh()` (for compatibility)
- Handles rotation state through Kalman filter

### 7. **Main DeepSort OBB Class** (`utils/deep_sort_obb.py`)
High-level interface:
```python
deepsort = DeepSortOBB(model_path, ...)
outputs = deepsort.update(obb_detections, confidences, frame)
```

## Usage Example

### Basic Integration

```python
from ultralytics.hara.hara_obb_deepsort.utils.deep_sort_obb import DeepSortOBB
from ultralytics.hara.hara_obb_deepsort.objdetector import Detector

# Initialize detector and tracker
detector = Detector()
deepsort = DeepSortOBB(model_path='path/to/reid_model.pt')

# In your main loop:
detections = detector.detect(frame)  # Returns OBB detections
confidences = [d[2] for d in detections]  # Extract confidence scores
obb_corners = [d[0] for d in detections]  # Extract corner points

# Track OBB detections
tracked_outputs = deepsort.update(obb_corners, confidences, frame)
# Returns: [[x1, y1, x2, y2, track_id], ...]
```

### With the Integration Layer

```python
from ultralytics.hara.hara_obb_deepsort import objtracker
from ultralytics.hara.hara_obb_deepsort.objdetector import Detector

detector = Detector()

# In your main loop:
frame, tracked_objects = objtracker.update(detector, frame)
```

## Configuration

### DeepSort Parameters

```python
deepsort = DeepSortOBB(
    model_path='path/to/reid_model.pt',  # ReID feature extractor
    max_dist=0.2,                         # Max cosine distance for matching
    min_confidence=0.3,                   # Min detection confidence
    max_iou_distance=0.7,                 # Max IOU distance threshold
    max_age=70,                           # Max frames without detection
    n_init=3,                             # Frames to confirm track
    use_rotated_features=True,            # Extract rotated image crops
)
```

### Feature Extraction

Two modes available:

**Mode 1: Rotated Features (Default)**
- Extracts crops aligned to OBB rotation
- Better representation but slower
- Enable with: `use_rotated_features=True`

**Mode 2: Axis-Aligned Features**
- Extracts axis-aligned bounding box regions
- Faster but loses rotation information
- Enable with: `use_rotated_features=False`

### IOU Matching

Two options:

**Option 1: Rotated IOU (Default)**
- Precise geometric intersection calculation
- Handles rotations properly
- Slightly slower

**Option 2: Axis-Aligned IOU**
- Faster fallback
- Uses bounding box approximation
- Less accurate for rotated objects

## Data Formats

### Input Detection Format
From YOLO OBB detector:
```python
detections = [
    (xyxyxyxy_corners, label, confidence),
    # where xyxyxyxy_corners is shape (4, 2) array of corner points
]
```

### Internal Representation
- **xyxyxyxy**: `[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]` - Corner points (4, 2)
- **xywhr**: `[cx, cy, w, h, angle]` - Center + dimensions + rotation

### Output Format
```python
outputs = [
    [x1, y1, x2, y2, track_id],  # Axis-aligned for compatibility
]
```

## Performance Tuning

### For Better Tracking Accuracy
1. Increase `n_init` (more frames to confirm track)
2. Use `use_rotated_features=True`
3. Lower `max_iou_distance` threshold
4. Use rotated IOU matching

### For Better Performance (Speed)
1. Decrease `n_init`
2. Use `use_rotated_features=False`
3. Use axis-aligned IOU matching via `iou_cost_fallback()`
4. Increase `max_age` to reduce computation

## Troubleshooting

### Import Errors
If you encounter `ModuleNotFoundError` when importing modules:
1. Ensure you're using proper relative imports with `utils.` prefix
2. Check that all files are in the `utils/` directory
3. Use absolute imports from the top-level package when needed

### Feature Extraction Failures
If rotated crop extraction fails:
- The system automatically falls back to axis-aligned crops
- Check that input images are valid numpy arrays
- Ensure corner points are within image bounds

### Tracking Instability
If tracks are jumping or being recreated frequently:
1. Increase `n_init` to require more detections before confirming
2. Increase `max_iou_distance` to allow larger positional changes
3. Lower `min_confidence` threshold if detections are too sparse
4. Check if detection quality is good

## Advanced Features

### Custom Kalman Filter State
The OBB Kalman filter tracks:
- Position: `cx, cy` (center coordinates)
- Size: `w, h` (width and height)
- Orientation: `angle` (rotation in degrees)
- Velocities: `vcx, vcy, vw, vh, vangle`

This allows predicting future object position AND orientation.

### Rotated IOU Calculation
Uses OpenCV's `cv2.rotatedRectangleIntersection()` for:
- Precise intersection area calculation
- Handles arbitrary rotation angles
- Falls back to axis-aligned IOU if it fails

### Angle Wrapping
Angles are normalized to [-180, 180] degrees to prevent:
- Discontinuities in angle differences
- Issues with angle averaging
- Kalman filter divergence

## File Structure

```
hara_obb_deepsort/
├── utils/
│   ├── __init__.py
│   ├── obb_utils.py              # Format conversions & utilities
│   ├── detection_obb.py          # OBB Detection class
│   ├── kalman_filter_obb.py      # Extended Kalman filter
│   ├── iou_matching_obb.py       # Rotated IOU matching
│   ├── deep_sort_obb.py          # Main OBB DeepSort class
│   ├── tracker_obb.py            # OBB-aware tracker
│   ├── track_obb.py              # OBB Track class
│   └── linear_assignment_obb.py  # OBB linear assignment
├── objdetector.py                # YOLO OBB detector wrapper
├── objtracker.py                 # Integration layer
├── demo.py                       # Demo script
├── test_obb_deepsort.py          # Test suite
└── README.md
```

## Testing

Run the test suite to verify installation:

```bash
python ultralytics/hara/hara_obb_deepsort/test_obb_deepsort.py
```

Expected output:
```
Testing OBB DeepSORT Implementation
=====================================
✓ OBB utils test passed
✓ OBB Detection test passed
✓ Kalman Filter OBB test passed

Results: 3/3 tests passed
🎉 All tests passed! OBB DeepSORT implementation is ready.
```

## References

### Related Projects
- **YOLOv8 OBB**: Object detection with oriented bounding boxes
- **DeepSORT**: Multi-object tracking with deep learning
- **OpenCV**: Computer vision functions for geometry

### Key Papers
- DeepSORT: Simple Online and Realtime Tracking with a Deep Association Metric
- Kalman Filter for tracking
- Hungarian Algorithm for bipartite matching

## Support

For issues or improvements:
1. Check the troubleshooting section
2. Review test cases for expected behavior
3. Examine import statements for correctness
4. Verify data formats match specifications

---

Last Updated: April 2026
Version: 1.0
Status: ✅ Complete and Tested

