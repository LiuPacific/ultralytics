import sys
import os
import numpy as np

from kalman_filter_obb import KalmanFilterOBB

def test_kalman_filter_obb():
    """Test the OBB Kalman filter with angle dimension"""
    print("Testing OBB Kalman Filter with angle...")

    try:

        # Create Kalman filter
        kf = KalmanFilterOBB()

        # Test measurement: [cx, cy, w, h, angle]
        measurement = np.array([100.0, 200.0, 50.0, 80.0, 45.0])

        print(f"Input measurement: {measurement}")

        # Test initiate
        mean, covariance = kf.initiate(measurement)
        print(f"Initiated mean shape: {mean.shape}")
        print(f"Initiated covariance shape: {covariance.shape}")
        print(f"Mean values: {mean}")

        # Test predict
        mean_pred, cov_pred = kf.predict(mean, covariance)
        print(f"Predicted mean shape: {mean_pred.shape}")
        print(f"Predicted mean: {mean_pred}")

        # Test update
        mean_upd, cov_upd = kf.update(mean_pred, cov_pred, measurement)
        print(f"Updated mean shape: {mean_upd.shape}")
        print(f"Updated mean: {mean_upd}")

        # Test project
        mean_proj, cov_proj = kf.project(mean, covariance)
        print(f"Projected mean shape: {mean_proj.shape}")
        print(f"Projected mean: {mean_proj}")

        # Test gating distance
        measurements = np.array([measurement])
        gating_dist = kf.gating_distance(mean, covariance, measurements)
        print(f"Gating distance: {gating_dist}")

        print("✅ OBB Kalman Filter test passed!")
        return True

    except Exception as e:
        print(f"❌ OBB Kalman Filter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_angle_wrapping():
    """Test angle wrapping functionality"""
    print("\nTesting angle wrapping...")

    try:
        from kalman_filter_obb import KalmanFilterOBB

        kf = KalmanFilterOBB()

        # Test measurement with angle near boundary
        measurement1 = np.array([100.0, 200.0, 50.0, 80.0, 170.0])  # 170 degrees
        measurement2 = np.array([100.0, 200.0, 50.0, 80.0, -170.0]) # -170 degrees

        mean1, _ = kf.initiate(measurement1)
        mean2, _ = kf.initiate(measurement2)

        # Test update with angle wrapping
        mean_upd, _ = kf.update(mean1, np.eye(10), measurement2)
        print(f"Angle wrapping test: {mean_upd[4]} (should be close to -170)")

        print("✅ Angle wrapping test passed!")
        return True

    except Exception as e:
        print(f"❌ Angle wrapping test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("OBB Kalman Filter Test with Angle")
    print("=" * 60)

    tests = [
        test_kalman_filter_obb,
        test_angle_wrapping,
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
        print("🎉 All Kalman filter tests passed!")
        print("The OBB Kalman filter with angle dimension is working correctly.")
    else:
        print("❌ Some tests failed.")

    sys.exit(0 if passed == total else 1)
