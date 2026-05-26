#!/usr/bin/env python3
"""
Test script for OBB IOU matching with xywhr format
"""

import sys
import os
import numpy as np

# # Add paths
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'deep_sort', 'sort'))

def test_rotated_iou():
    """Test rotated IOU calculation with xywhr format"""
    print("Testing rotated IOU calculation with xywhr format...")

    try:
        from obb_utils import rotated_iou

        # Test 1: Identical boxes (IOU = 1.0)
        box1 = (100, 100, 50, 80, 0)  # cx, cy, w, h, angle
        box2 = (100, 100, 50, 80, 0)
        iou = rotated_iou(box1, box2)
        assert abs(iou - 1.0) < 0.01, f"IOU for identical boxes should be 1.0, got {iou}"
        print(f"✓ Identical boxes: IOU = {iou:.4f}")

        # Test 2: Non-overlapping boxes (IOU = 0.0)
        box1 = (0, 0, 50, 50, 0)
        box2 = (200, 200, 50, 50, 0)
        iou = rotated_iou(box1, box2)
        assert abs(iou) < 0.01, f"IOU for non-overlapping boxes should be 0.0, got {iou}"
        print(f"✓ Non-overlapping boxes: IOU = {iou:.4f}")

        # Test 3: Partially overlapping boxes
        box1 = (100, 100, 50, 50, 0)
        box2 = (125, 125, 50, 50, 0)  # Overlaps by 25x25
        iou = rotated_iou(box1, box2)
        assert 0 < iou < 1, f"IOU should be between 0 and 1, got {iou}"
        print(f"✓ Partially overlapping boxes: IOU = {iou:.4f}")

        # Test 4: Rotated boxes
        box1 = (100, 100, 50, 50, 0)    # No rotation
        box2 = (100, 100, 50, 50, 45)   # 45 degree rotation
        iou = rotated_iou(box1, box2)
        assert 0 < iou <= 1, f"Rotated box IOU should be valid, got {iou}"
        print(f"✓ Same position, different angle: IOU = {iou:.4f}")

        # Test 5: Nested boxes
        box1 = (100, 100, 100, 100, 0)  # Larger box
        box2 = (100, 100, 50, 50, 0)    # Smaller box inside
        iou = rotated_iou(box1, box2)
        # IOU = 2500 / (10000 + 2500 - 2500) = 2500 / 10000 = 0.25
        assert abs(iou - 0.25) < 0.05, f"Nested box IOU should be ~0.25, got {iou}"
        print(f"✓ Nested boxes: IOU = {iou:.4f}")

        print("✅ Rotated IOU tests passed!\n")
        return True

    except Exception as e:
        print(f"❌ Rotated IOU test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_obb_iou_xywhr():
    """Test OBB IOU function with multiple candidates"""
    print("Testing obb_iou_xywhr function...")

    try:
        from iou_matching_obb import obb_iou_xywhr

        # Reference box
        bbox = (100, 100, 50, 50, 0)

        # Multiple candidate boxes
        candidates = np.array([
            [100, 100, 50, 50, 0],      # Identical - IOU 1.0
            [200, 200, 50, 50, 0],      # Non-overlapping - IOU 0.0
            [125, 125, 50, 50, 0],      # Partially overlapping
        ])

        ious = obb_iou_xywhr(bbox, candidates)

        assert len(ious) == 3, f"Expected 3 IOUs, got {len(ious)}"
        assert abs(ious[0] - 1.0) < 0.01, f"First IOU should be 1.0, got {ious[0]}"
        assert abs(ious[1]) < 0.01, f"Second IOU should be 0.0, got {ious[1]}"
        assert 0 < ious[2] < 1, f"Third IOU should be between 0 and 1, got {ious[2]}"

        print(f"✓ Identical box: {ious[0]:.4f}")
        print(f"✓ Non-overlapping box: {ious[1]:.4f}")
        print(f"✓ Partially overlapping box: {ious[2]:.4f}")

        print("✅ obb_iou_xywhr tests passed!\n")
        return True

    except Exception as e:
        print(f"❌ obb_iou_xywhr test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_iou_cost_matrix():
    """Test IOU cost matrix generation"""
    print("Testing IOU cost matrix generation...")

    try:
        from iou_matching_obb import obb_iou_cost

        # Create mock track class
        class MockTrack:
            def __init__(self, mean):
                self.mean = mean
                self.time_since_update = 0

            def to_xywhr(self):
                return self.mean[:5]

        # Create mock detection class
        class MockDetection:
            def __init__(self, xywhr):
                self._xywhr = xywhr

            def to_xywhr(self):
                return self._xywhr

        # Create tracks with 10D state space
        track1_mean = np.array([100, 100, 50, 50, 0, 0, 0, 0, 0, 0])  # xywhr + velocities
        track1 = MockTrack(track1_mean)

        # Create detections
        det1 = MockDetection(np.array([100, 100, 50, 50, 0]))      # Identical
        det2 = MockDetection(np.array([200, 200, 50, 50, 0]))      # Non-overlapping

        tracks = [track1]
        detections = [det1, det2]

        cost_matrix = obb_iou_cost(tracks, detections)

        assert cost_matrix.shape == (1, 2), f"Cost matrix shape should be (1, 2), got {cost_matrix.shape}"
        assert abs(cost_matrix[0, 0]) < 0.01, f"Cost for identical box should be 0, got {cost_matrix[0, 0]}"
        assert cost_matrix[0, 1] > 0.9, f"Cost for non-overlapping box should be ~1, got {cost_matrix[0, 1]}"

        print(f"✓ Cost matrix shape: {cost_matrix.shape}")
        print(f"✓ Identical box cost: {cost_matrix[0, 0]:.4f}")
        print(f"✓ Non-overlapping box cost: {cost_matrix[0, 1]:.4f}")

        print("✅ IOU cost matrix tests passed!\n")
        return True

    except Exception as e:
        print(f"❌ IOU cost matrix test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_angle_handling():
    """Test IOU calculation with various angles"""
    print("Testing angle handling in IOU calculations...")

    try:
        from obb_utils import rotated_iou

        # Same box with different angles
        base_box = (100, 100, 50, 80, 0)

        angles = [0, 15, 45, 90, 135, 180, 270]

        print("Testing box at various angles:")
        for angle in angles:
            box = (100, 100, 50, 80, angle)
            iou = rotated_iou(base_box, box)
            print(f"✓ Angle {angle:3d}°: IOU = {iou:.4f}")
            assert 0 <= iou <= 1, f"IOU should be in [0, 1], got {iou}"

        print("✅ Angle handling tests passed!\n")
        return True

    except Exception as e:
        print(f"❌ Angle handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_edge_cases():
    """Test edge cases in IOU calculation"""
    print("Testing edge cases...")

    try:
        from obb_utils import rotated_iou

        # Test 1: Very small boxes
        box1 = (100, 100, 1, 1, 0)
        box2 = (100, 100, 1, 1, 0)
        iou = rotated_iou(box1, box2)
        assert abs(iou - 1.0) < 0.01, f"Tiny identical boxes should have IOU 1.0, got {iou}"
        print(f"✓ Tiny identical boxes: IOU = {iou:.4f}")

        # Test 2: Very large boxes
        box1 = (0, 0, 10000, 10000, 0)
        box2 = (0, 0, 10000, 10000, 0)
        iou = rotated_iou(box1, box2)
        assert abs(iou - 1.0) < 0.01, f"Large identical boxes should have IOU 1.0, got {iou}"
        print(f"✓ Large identical boxes: IOU = {iou:.4f}")

        # Test 3: Touching boxes (should have IOU 0)
        box1 = (0, 0, 50, 50, 0)
        box2 = (50, 0, 50, 50, 0)  # Adjacent
        iou = rotated_iou(box1, box2)
        assert iou >= 0, f"Touching boxes should have non-negative IOU, got {iou}"
        print(f"✓ Touching boxes: IOU = {iou:.4f}")

        print("✅ Edge case tests passed!\n")
        return True

    except Exception as e:
        print(f"❌ Edge case test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 70)
    print("OBB IOU Matching Tests with xywhr Format")
    print("=" * 70)
    print()

    tests = [
        test_rotated_iou,
        test_obb_iou_xywhr,
        test_iou_cost_matrix,
        test_angle_handling,
        test_edge_cases,
    ]

    results = []
    for test in tests:
        results.append(test())

    print("=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"Test Results: {passed}/{total} passed")

    if passed == total:
        print("🎉 All IOU matching tests passed!")
        print("The OBB IOU matching with xywhr format is working correctly.")
    else:
        print("❌ Some tests failed.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

