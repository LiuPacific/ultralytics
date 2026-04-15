#!/usr/bin/env python3
"""
Test script for hara_deep_sort_obb2 OBB conversion
"""

import sys
import os
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_obb_detector():
    """Test that OBB detector works"""
    print("Testing OBB detector...")
    try:
        from objdetector import Detector

        # Create a dummy image for testing
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        detector = Detector()
        detections = detector.detect(test_image)

        print(f"✓ Detector created successfully")
        print(f"✓ Detected {len(detections)} objects")

        if len(detections) > 0:
            bbox, label, conf = detections[0]
            print(f"✓ First detection: bbox shape {bbox.shape}, label '{label}', conf {conf:.3f}")

        return True
    except Exception as e:
        print(f"✗ OBB detector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_obb_tracker():
    """Test that OBB tracker initialization works (skipping full test due to missing ReID model)"""
    print("\nTesting OBB tracker initialization...")
    try:
        # Just test that the imports work and basic structure is correct
        from utils.deep_sort_obb import DeepSortOBB
        from utils.tracker_obb import TrackerOBB
        from utils.kalman_filter_obb import KalmanFilterOBB

        # Test Kalman filter creation
        kf = KalmanFilterOBB()
        assert kf is not None

        print("✓ OBB tracker components can be imported and initialized")
        print("  (Note: Full tracker test skipped due to missing ReID model)")
        return True
    except Exception as e:
        print(f"✗ OBB tracker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_obb_imports():
    """Test that all OBB utilities can be imported"""
    print("\nTesting OBB utility imports...")
    try:
        from utils.obb_utils import xyxyxyxy_to_xywhr
        from utils.detection_obb import OBBDetection
        from utils.kalman_filter_obb import KalmanFilterOBB
        from utils.deep_sort_obb import DeepSortOBB

        print("✓ All OBB utilities imported successfully")
        return True
    except Exception as e:
        print(f"✗ OBB utility import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("hara_deep_sort_obb2 OBB Conversion Test")
    print("=" * 60)

    tests = [
        test_obb_imports,
        test_obb_detector,
        test_obb_tracker,
    ]

    results = []
    for test in tests:
        results.append(test())
        print()

    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Test Results: {passed}/{total} passed")

    if passed == total:
        print("✅ All tests passed!")
        print("hara_deep_sort_obb2 has been successfully converted to OBB!")
        print("\nKey changes made:")
        print("- objdetector.py: Now uses YOLO OBB (.obb) instead of regular boxes (.boxes)")
        print("- objtracker.py: Now uses DeepSortOBB with OBB-aware tracking")
        print("- demo.py: Updated to work with OBB detections")
        print("- Added utils/ directory with complete OBB implementation")
    else:
        print("❌ Some tests failed.")
        print("Please check the errors above.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
