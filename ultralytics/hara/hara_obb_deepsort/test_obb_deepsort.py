#!/usr/bin/env python3
"""
Test script for OBB DeepSORT implementation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_obb_utils():
    """Test OBB utility functions"""
    try:
        from utils.obb_utils import xyxyxyxy_to_xywhr, xywhr_to_xyxyxyxy, obb_to_xyxy_aligned

        # Test conversion functions
        test_obb = [[100, 100], [150, 100], [150, 200], [100, 200]]  # Simple rectangle
        cx, cy, w, h, angle = xyxyxyxy_to_xywhr(test_obb)
        print(f"Original OBB: {test_obb}")
        print(f"Converted to xywhr: cx={cx:.1f}, cy={cy:.1f}, w={w:.1f}, h={h:.1f}, angle={angle:.1f}")

        # Test reverse conversion
        reconstructed = xywhr_to_xyxyxyxy(cx, cy, w, h, angle)
        print(f"Reconstructed OBB: {reconstructed}")

        # Test axis-aligned conversion
        x1, y1, x2, y2 = obb_to_xyxy_aligned(test_obb)
        print(f"Axis-aligned bbox: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

        print("✓ OBB utils test passed")
        return True
    except Exception as e:
        print(f"✗ OBB utils test failed: {e}")
        return False

def test_obb_detection():
    """Test OBB Detection class"""
    try:
        from utils.detection_obb import OBBDetection
        from utils.obb_utils import xyxyxyxy_to_xywhr

        test_obb = [[100, 100], [150, 100], [150, 200], [100, 200]]
        detection = OBBDetection(test_obb, 0.95, [0.1, 0.2, 0.3])

        print(f"OBB Detection created: {detection}")
        print(f"xywhr: {detection.to_xywhr()}")
        print(f"xyxyxyxy: {detection.to_xyxyxyxy()}")
        print(f"xyah: {detection.to_xyah()}")

        print("✓ OBB Detection test passed")
        return True
    except Exception as e:
        print(f"✗ OBB Detection test failed: {e}")
        return False

def test_kalman_filter():
    """Test OBB Kalman Filter"""
    try:
        from utils.kalman_filter_obb import KalmanFilterOBB

        kf = KalmanFilterOBB()
        measurement = [125, 150, 50, 100, 0]  # cx, cy, w, h, angle

        mean, covariance = kf.initiate(measurement)
        print(f"Kalman filter initiated with mean shape: {mean.shape}")
        print(f"Covariance shape: {covariance.shape}")

        # Test prediction
        mean_pred, cov_pred = kf.predict(mean, covariance)
        print(f"Prediction successful, mean shape: {mean_pred.shape}")

        # Test update
        mean_upd, cov_upd = kf.update(mean_pred, cov_pred, measurement)
        print(f"Update successful, mean shape: {mean_upd.shape}")

        print("✓ Kalman Filter OBB test passed")
        return True
    except Exception as e:
        print(f"✗ Kalman Filter OBB test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing OBB DeepSORT Implementation")
    print("=" * 40)

    tests = [
        test_obb_utils,
        test_obb_detection,
        test_kalman_filter,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! OBB DeepSORT implementation is ready.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
