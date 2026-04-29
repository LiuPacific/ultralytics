import numpy as np
from scipy.optimize import linear_sum_assignment


def select_15_points_by_distance_and_confidence(
        prev_points,
        curr_points,
        curr_scores,
        alpha_distance=0.7,
        alpha_score=0.3,
        x_min=470,
        x_max=1900,
        y_min=670,
        y_max=1600,
):
    """
    Select 15 points from current detections by matching them to previous 15 points.

    Parameters
    ----------
    prev_points : np.ndarray
        Shape: (15, 2)
        Points from previous frame/image. Each row is [x, y].

    curr_points : np.ndarray
        Shape: (m, 2)
        Points from current frame/image. Each row is [x, y].

    curr_scores : np.ndarray
        Shape: (m,)
        Confidence scores of current points, values from 0 to 1.

    alpha_distance : float
        Weight for distance cost. Larger means movement smoothness is more important.

    alpha_score : float
        Weight for confidence reward. Larger means confidence score is more important.

    Returns
    -------
    selected_points : np.ndarray
        Shape: (15, 2), selected current-frame points.

    selected_scores : np.ndarray
        Shape: (15,), confidence scores of selected points.

    matched_pairs : list
        List of tuples: (prev_index, curr_index, distance, score, cost)
    """

    prev_points = np.asarray(prev_points, dtype=float)
    curr_points = np.asarray(curr_points, dtype=float)
    curr_scores = np.asarray(curr_scores, dtype=float)

    if prev_points.shape != (15, 2):
        raise ValueError("prev_points must have shape (15, 2).")

    if curr_points.ndim != 2 or curr_points.shape[1] != 2:
        raise ValueError("curr_points must have shape (m, 2).")

    if curr_scores.shape[0] != curr_points.shape[0]:
        raise ValueError("curr_scores length must match curr_points length.")

    if curr_points.shape[0] < 15:
        raise ValueError("curr_points must contain at least 15 points.")

    # Maximum possible distance in your valid image region
    d_max = np.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)

    # Distance matrix, shape: (15, m)
    # distances[i, j] = distance between previous point i and current point j
    diff = prev_points[:, None, :] - curr_points[None, :, :]
    distances = np.linalg.norm(diff, axis=2)

    # Normalize distance to roughly 0-1
    norm_distances = distances / d_max

    # Cost matrix:
    # lower distance is better, higher score is better
    cost_matrix = alpha_distance * norm_distances - alpha_score * curr_scores[None, :]

    # Hungarian algorithm
    prev_indices, curr_indices = linear_sum_assignment(cost_matrix)

    selected_points = curr_points[curr_indices]
    selected_scores = curr_scores[curr_indices]

    matched_pairs = []
    for i, j in zip(prev_indices, curr_indices):
        matched_pairs.append(
            {
                "prev_index": int(i),
                "curr_index": int(j),
                "distance": float(distances[i, j]),
                "score": float(curr_scores[j]),
                "cost": float(cost_matrix[i, j]),
            }
        )

    return selected_points, selected_scores, matched_pairs, curr_indices



prev_points = np.array([
    [500, 800],
    [600, 820],
    [700, 850],
    [800, 900],
    [900, 920],
    [1000, 950],
    [1100, 980],
    [1200, 1000],
    [1300, 1050],
    [1400, 1100],
    [1500, 1150],
    [1600, 1200],
    [1700, 1250],
    [1800, 1300],
    [1850, 1350],
])

curr_points = np.array([
    [610, 825],
    [705, 845],
    [810, 910],
    [895, 925],
    [1005, 955],
    [1110, 985],
    [1195, 995],
    [1310, 1045],
    [1395, 1110],
    [1510, 1160],
    [1595, 1195],
    [1710, 1260],
    [1790, 1310],
    [1840, 1340],
    [505, 805],
    [300, 300],      # noisy extra detection
])

curr_scores = np.array([
    0.91, 0.88, 0.95, 0.90, 0.87,
    0.92, 0.89, 0.93, 0.86, 0.90,
    0.94, 0.88, 0.91, 0.89, 0.92,
    0.80,            # noisy point has confidence, but is far away
])

selected_points, selected_scores, matched_pairs, curr_indices = select_15_points_by_distance_and_confidence(
    prev_points,
    curr_points,
    curr_scores,
    alpha_distance=0.7,
    alpha_score=0.3,
)

print("Selected indices:")
print(curr_indices)


print("Selected points:")
print(selected_points)

print("\nSelected scores:")
print(selected_scores)

print("\nMatched pairs:")
for pair in matched_pairs:
    print(pair)