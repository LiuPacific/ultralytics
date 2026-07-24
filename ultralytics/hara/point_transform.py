import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ==============================================================================
# 1. Thin-Plate Spline (TPS) Model
# (This is the same robust TPS implementation from the previous report)
# ==============================================================================
class ThinPlateSpline:
    """
    A robust Thin-Plate Spline (TPS) implementation that finds a non-rigid
    transformation between two sets of control points.
    """
    def __init__(self, alpha=0.0):
        self.alpha = alpha
        self.src_pts = None
        self.coeffs = None

    def _tps_kernel(self, r):
        # The TPS radial basis function: U(r) = r^2 * log(r)
        # A small epsilon is added to avoid log(0) -> -inf.
        r_safe = r.clone()
        r_safe[r == 0] = 1e-9
        return r_safe**2 * torch.log(r_safe)

    def fit(self, src_pts, dst_pts):
        if src_pts.shape[0] != dst_pts.shape[0]:
            raise ValueError("Source and destination must have the same number of points.")
        if src_pts.shape[0] < 3:
            raise ValueError("At least 3 control points are required for TPS.")

        self.src_pts = src_pts
        n_pts, n_dims = src_pts.shape
        device = src_pts.device
        dtype = src_pts.dtype

        # Compute pairwise distances to form the K matrix
        dist_matrix = torch.cdist(src_pts, src_pts, p=2)
        K = self._tps_kernel(dist_matrix)

        # Add regularization to the diagonal
        if self.alpha > 0:
            K.diagonal().add_(self.alpha)

        # Construct the P matrix for the affine part
        P = torch.cat([torch.ones(n_pts, 1, device=device, dtype=dtype), src_pts], dim=1)

        # Construct the full L matrix
        L_upper = torch.cat([K, P], dim=1)
        L_lower = torch.cat([P.T, torch.zeros(n_dims + 1, n_dims + 1, device=device, dtype=dtype)], dim=1)
        L = torch.cat([L_upper, L_lower], dim=0)

        # Construct the Y matrix
        Y = torch.cat([dst_pts, torch.zeros(n_dims + 1, n_dims, device=device, dtype=dtype)], dim=0)

        # Solve the linear system L * W = Y for the coefficients W
        self.coeffs = torch.linalg.solve(L, Y)

    def transform(self, points):
        if self.coeffs is None:
            raise RuntimeError("The model has not been fitted yet. Call fit() first.")

        n_pts, n_dims = points.shape
        device = points.device
        dtype = points.dtype

        # Compute distances from new points to source control points
        dist_matrix = torch.cdist(points, self.src_pts, p=2)
        K_new = self._tps_kernel(dist_matrix)

        # Construct the affine part for the new points
        P_new = torch.cat([torch.ones(n_pts, 1, device=device, dtype=dtype), points], dim=1)

        # Combine into the full matrix for transformation
        L_new = torch.cat([K_new, P_new], dim=1)

        # Apply the coefficients to get the transformed points
        transformed_points = torch.matmul(L_new, self.coeffs)

        return transformed_points

# ==============================================================================
# 2. Multi-Layer Perceptron (MLP) Model
# ==============================================================================
class CoordinateMLP(nn.Module):
    """
    A simple Multi-Layer Perceptron for learning a 2D coordinate transformation.
    """
    def __init__(self, input_dim=2, output_dim=2, hidden_dim=256):
        super(CoordinateMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)

# ==============================================================================
# 3. The Optimized, Resolution-Aware Registrar
# ==============================================================================
class ResolutionAwareRegistrar:
    """
    A class to handle image registration that explicitly accounts for different
    image resolutions by normalizing coordinates before transformation.
    """
    def __init__(self, src_resolution, dst_resolution):
        """
        Initializes the registrar with the resolutions of the source and
        destination images.

        Args:
            src_resolution (tuple): (width, height) of the source image.
            dst_resolution (tuple): (width, height) of the target image.
        """
        # Store resolutions as tensors for calculations
        self.src_res = torch.tensor([src_resolution[0] - 1, src_resolution[1] - 1], dtype=torch.float32)
        self.dst_res = torch.tensor([dst_resolution[0] - 1, dst_resolution[1] - 1], dtype=torch.float32)
        self.model = None
        self.method = None

    def _normalize(self, points, resolution):
        """Normalizes pixel coordinates to the [0, 1] range."""
        return points / resolution

    def _denormalize(self, points_norm, resolution):
        """De-normalizes coordinates from [0, 1] back to pixel coordinates."""
        return points_norm * resolution

    def fit(self, src_pts, dst_pts, method='tps', **kwargs):
        """
        Fits a transformation model (TPS or MLP) to the control points.

        Args:
            src_pts (array-like): (N, 2) array of source pixel coordinates.
            dst_pts (array-like): (N, 2) array of destination pixel coordinates.
            method (str): 'tps' or 'mlp'.
            **kwargs: Additional arguments for the model (e.g., alpha for TPS,
                      epochs/lr for MLP).
        """
        self.method = method.lower()
        src_pts_t = torch.as_tensor(src_pts, dtype=torch.float32)
        dst_pts_t = torch.as_tensor(dst_pts, dtype=torch.float32)

        # --- CRITICAL STEP: Normalize coordinates before fitting ---
        src_pts_norm = self._normalize(src_pts_t, self.src_res)
        dst_pts_norm = self._normalize(dst_pts_t, self.dst_res)

        if self.method == 'tps':
            alpha = kwargs.get('alpha', 1e-4)
            self.model = ThinPlateSpline(alpha=alpha)
            self.model.fit(src_pts_norm, dst_pts_norm)
            print("TPS model fitted successfully on normalized coordinates.")

        elif self.method == 'mlp':
            hidden_dim = kwargs.get('hidden_dim', 256)
            epochs = kwargs.get('epochs', 20000)
            lr = kwargs.get('lr', 1e-4)

            self.model = CoordinateMLP(hidden_dim=hidden_dim)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=lr)

            print(f"Training MLP for {epochs} epochs...")
            for epoch in range(epochs):
                self.model.train()
                optimizer.zero_grad()
                outputs = self.model(src_pts_norm)
                loss = criterion(outputs, dst_pts_norm)
                loss.backward()
                optimizer.step()

                if (epoch + 1) % 5000 == 0:
                    print(f'  Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.8f}')
            print("MLP model trained successfully on normalized coordinates.")

        else:
            raise ValueError(f"Method '{self.method}' is not supported. Choose 'tps' or 'mlp'.")

    def transform(self, points):
        """
        Transforms new points from the source to the target coordinate system.

        Args:
            points (array-like): (M, 2) array of source pixel coordinates to transform.

        Returns:
            Tensor of transformed points in the target pixel coordinate system.
        """
        if self.model is None:
            raise RuntimeError("The registrar has not been fitted yet. Call fit() first.")

        points_t = torch.as_tensor(points, dtype=torch.float32)
        if points_t.dim() == 1:
            points_t = points_t.unsqueeze(0)

        # --- CRITICAL STEP: Apply the full normalization -> transform -> de-normalization pipeline ---
        points_norm = self._normalize(points_t, self.src_res)

        if self.method == 'mlp':
            self.model.eval()
            with torch.no_grad():
                transformed_norm = self.model(points_norm)
        else: # TPS
            transformed_norm = self.model.transform(points_norm)

        transformed_pixels = self._denormalize(transformed_norm, self.dst_res)

        return transformed_pixels

# ==============================================================================
# 4. Example Usage
# ==============================================================================
if __name__ == '__main__':
    # --- Configuration ---
    # Define the resolutions of your two images
    # Example: A high-res RGB image and a lower-res Thermal image
    SOURCE_IMAGE_RESOLUTION = (2336, 1752)  # (width, height)
    TARGET_IMAGE_RESOLUTION = (640, 480)   # (width, height)

    # Manually selected control points in PIXEL COORDINATES
    # These should be identifiable features in both images (e.g., corners of the feeder)
    # IMPORTANT: The order of points must correspond between the two arrays.
    source_control_points = np.array([
        [525, 1437],
        [739, 1360],
        [1040, 1433],
        [944, 1353],
        [837, 795],
        [666, 1333]
    ], dtype=np.float32)

    # Corresponding points in the lower-resolution thermal image
    target_control_points = np.array([
        [120, 415],
        [185, 398],
        [285, 420],
        [249, 394],
        [208, 211],
        [160, 384]
    ], dtype=np.float32)

    # A new point in the SOURCE (RGB) image that we want to find in the TARGET (thermal) image
    point_to_transform = np.array([993, 1265], dtype=np.float32)

    print("="*50)
    print("Method 1: Thin-Plate Spline (TPS) Registration")
    print("="*50)

    # 1. Initialize the registrar with the image resolutions
    registrar_tps = ResolutionAwareRegistrar(
        src_resolution=SOURCE_IMAGE_RESOLUTION,
        dst_resolution=TARGET_IMAGE_RESOLUTION
    )

    # 2. Fit the TPS model using the pixel-based control points
    registrar_tps.fit(source_control_points, target_control_points, method='tps', alpha=1e-5)

    # 3. Transform the new point
    transformed_point_tps = registrar_tps.transform(point_to_transform)
    print(f"\nOriginal point (in {SOURCE_IMAGE_RESOLUTION} image): {point_to_transform}")
    print(f"Transformed point (in {TARGET_IMAGE_RESOLUTION} image) via TPS: {transformed_point_tps.squeeze().numpy()}\n\n")
    # Original point (in (2336, 1752) image): [ 993. 1265.]
    # Transformed point (in (640, 480) image) via TPS: [262.28915 364.70807], in fact, it's 267,365

    print("="*50)
    print("Method 2: Multi-Layer Perceptron (MLP) Registration")
    print("="*50)

    # 1. Initialize a new registrar for the MLP
    registrar_mlp = ResolutionAwareRegistrar(
        src_resolution=SOURCE_IMAGE_RESOLUTION,
        dst_resolution=TARGET_IMAGE_RESOLUTION
    )

    # 2. Fit the MLP model
    registrar_mlp.fit(
        source_control_points,
        target_control_points,
        method='mlp',
        epochs=25000,
        lr=1e-4,
        hidden_dim=256
    )

    # 3. Transform the new point
    transformed_point_mlp = registrar_mlp.transform(point_to_transform)
    print(f"\nOriginal point (in {SOURCE_IMAGE_RESOLUTION} image): {point_to_transform}")
    print(f"Transformed point (in {TARGET_IMAGE_RESOLUTION} image) via MLP: {transformed_point_mlp.squeeze().numpy()}")
    # Original point (in (2336, 1752) image): [ 993. 1265.]
    # Transformed point (in (640, 480) image) via MLP: [257.1639  364.06406], in fact, it's 267,365
