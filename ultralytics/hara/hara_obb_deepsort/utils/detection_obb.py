import os
print(os.getcwd())

import numpy as np
from .obb_utils import xyxyxyxy_to_xywhr, xywhr_to_xyxyxyxy, obb_to_xyxy_aligned


class OBBDetection(object):
    """
    This class represents an oriented bounding box (OBB) detection in a single image.

    Parameters
    ----------
    xyxyxyxy : array_like
        Oriented bounding box corner points in format `(x1,y1, x2,y2, x3,y3, x4,y4)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.

    Attributes
    ----------
    xyxyxyxy : ndarray
        Oriented bounding box corner points in format `(x1,y1, x2,y2, x3,y3, x4,y4)`.
    xywhr : ndarray
        Oriented bounding box in format `(center_x, center_y, width, height, angle)`.
    confidence : ndarray
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.
    """

    def __init__(self, xyxyxyxy, confidence, feature):
        self.xyxyxyxy = np.asarray(xyxyxyxy, dtype=float).reshape(4, 2)
        self.xywhr = np.asarray(xyxyxyxy_to_xywhr(self.xyxyxyxy), dtype=float)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32) if feature is not None else None

    def to_xyxyxyxy(self):
        """Get bounding box corner points in format `(x1,y1, x2,y2, x3,y3, x4,y4)`."""
        return self.xyxyxyxy.copy()

    def to_xywhr(self):
        """Get bounding box in format `(center_x, center_y, width, height, angle)`."""
        return self.xywhr.copy()

    def to_xyxy_aligned(self):
        """Convert OBB to axis-aligned bounding box format `(x1, y1, x2, y2)` for fallback."""
        return obb_to_xyxy_aligned(self.xyxyxyxy)

    def to_tlwh_aligned(self):
        """Convert OBB to axis-aligned tlwh format `(top_left_x, top_left_y, width, height)` for compatibility."""
        x1, y1, x2, y2 = self.to_xyxy_aligned()
        return np.array([x1, y1, x2 - x1, y2 - y1], dtype=float)

    def to_xyah(self):
        """Convert OBB to format `(center x, center y, aspect ratio, height)` for compatibility."""
        # Convert xywhr to xyah format (axis-aligned approximation)
        cx, cy, w, h, angle = self.xywhr
        aspect_ratio = w / h if h > 0 else 0
        return np.array([cx, cy, aspect_ratio, h], dtype=float)

    def get_centroid(self):
        """Get the centroid (center) of the OBB."""
        return self.xywhr[:2].copy()

    def get_angle(self):
        """Get the rotation angle of the OBB in degrees."""
        return self.xywhr[4]

    def get_dimensions(self):
        """Get the width and height of the OBB."""
        return self.xywhr[2:4].copy()

    @classmethod
    def from_xywhr(cls, xywhr, confidence, feature):
        """Create OBBDetection from xywhr format."""
        xyxyxyxy = xywhr_to_xyxyxyxy(*xywhr)
        return cls(xyxyxyxy, confidence, feature)

    def __repr__(self):
        return f"OBBDetection(xywhr={self.xywhr}, confidence={self.confidence:.3f})"
