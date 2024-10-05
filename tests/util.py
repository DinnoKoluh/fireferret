import numpy as np


def generate_random_cluster_points(point: np.ndarray, num_points: int = 10, max_distance: float = 10):
    N = point.shape[0]  # Dimensionality of the space

    # Generate M random unit vectors (directions)
    directions = np.random.randn(num_points, N)
    directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]  # Normalize to unit vectors

    # Generate M random distances between 0 and D
    distances = np.random.uniform(0, max_distance, num_points)

    # Generate random points by scaling directions with distances
    random_cluster = point + directions * distances[:, np.newaxis]

    return random_cluster
