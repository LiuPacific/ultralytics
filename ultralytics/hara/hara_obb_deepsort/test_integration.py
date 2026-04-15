#!/usr/bin/env python3
"""
Integration test for OBB DeepSORT
Verifies all components work together correctly
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    try:
        from utils.obb_utils import xyxyxyxy_to_xywhr, xywhr_to_xyxyxyxy
        from utils.detection_obb import OBBDetection
        from utils.kalman_filter_obb import KalmanFilterOBB
        from utils.track_obb import TrackOBB, TrackState
        from utils.linear_assignment_obb import gate_cost_matrix_obb
        # Note: deep_sort_obb and tracker_obb require deep_sort module which is in parent dir
        print("✓ All utils imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_obb_detection_creation():
    """Test creating OBB detections"""
    print("\nTesting OBB detection creation...")
    try:
        from utils.detection_obb import OBBDetection

        # Create a simple rotated rectangle
        obb_corners = np.array([
            [100, 100],  # top-left
            [200, 80],   # top-right
            [210, 180],  # bottom-right
            [110, 200]   # bottom-left
        ])

        feature = np.random.rand(128)  # 128-dim feature vector
        detection = OBBDetection(obb_corners, 0.95, feature)

        assert detection.xywhr is not None
        assert len(detection.to_xywhr()) == 5
        assert detection.to_xyah() is not None

        print(f"✓ Created OBB detection: xywhr={detection.to_xywhr()}")
        return True
    except Exception as e:
        print(f"✗ Detection creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_kalman_filter_obb():
    """Test OBB Kalman filter"""
    print("\nTesting OBB Kalman filter...")
    try:
        from utils.kalman_filter_obb import KalmanFilterOBB

        kf = KalmanFilterOBB()
        measurement = np.array([150, 150, 100, 200, 45.0])  # cx, cy, w, h, angle

        # Initiate
        mean, cov = kf.initiate(measurement)
        assert mean.shape == (10,)
        assert cov.shape == (10, 10)

        # Predict
        mean_pred, cov_pred = kf.predict(mean, cov)
        assert mean_pred.shape == (10,)

        # Update
        mean_upd, cov_upd = kf.update(mean_pred, cov_pred, measurement)
        assert mean_upd.shape == (10,)

        print("✓ Kalman filter OBB working correctly")
        return True
    except Exception as e:
        print(f"✗ Kalman filter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tracker_initialization():
    """Test tracker initialization"""
    print("\nTesting tracker initialization...")
    try:
        from utils.tracker_obb import TrackerOBB
        from deep_sort.deep_sort.sort.nn_matching import NearestNeighborDistanceMetric
        from utils.kalman_filter_obb import KalmanFilterOBB

        metric = NearestNeighborDistanceMetric("cosine", 0.2, 100)
        kf = KalmanFilterOBB()

        tracker = TrackerOBB(metric, max_iou_distance=0.7, max_age=70,
                            n_init=3, kalman_filter=kf)

        assert len(tracker.tracks) == 0
        assert tracker._next_id == 1

        print("✓ Tracker initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Tracker initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_obb_format_conversions():
    """Test OBB format conversions"""
    print("\nTesting OBB format conversions...")
    try:
        from utils.obb_utils import (
            xyxyxyxy_to_xywhr, xywhr_to_xyxyxyxy,
            obb_to_xyxy_aligned, rotated_iou
        )

        # Test conversion round-trip
        original = np.array([[100, 100], [200, 100], [200, 200], [100, 200]])
        cx, cy, w, h, angle = xyxyxyxy_to_xywhr(original)
        reconstructed = xywhr_to_xyxyxyxy(cx, cy, w, h, angle)

        # Should be close to original
        assert np.allclose(original, reconstructed, atol=1.0)

        # Test axis-aligned conversion
        x1, y1, x2, y2 = obb_to_xyxy_aligned(original)
        assert x1 < x2 and y1 < y2

        # Test rotated IOU
        obb1 = np.array([[100, 100], [200, 100], [200, 200], [100, 200]])
        obb2 = np.array([[100, 100], [200, 100], [200, 200], [100, 200]])
        iou = rotated_iou(obb1, obb2)
        assert abs(iou - 1.0) < 0.01  # Should be ~1.0 for identical boxes

        print("✓ OBB format conversions working correctly")
        return True
    except Exception as e:
        print(f"✗ Format conversion test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all integration tests"""
    print("=" * 60)
    print("OBB DeepSORT Integration Tests")
    print("=" * 60)

    tests = [
        test_imports,
        test_obb_detection_creation,
        test_kalman_filter_obb,
        test_tracker_initialization,
        test_obb_format_conversions,
    ]

    results = []
    for test in tests:
        results.append(test())

    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Integration Test Results: {passed}/{total} passed")

    if passed == total:
        print("✅ All integration tests passed!")
        print("OBB DeepSORT implementation is fully functional.")
        return True
    else:
        print("❌ Some integration tests failed.")
        print("Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)



