# https://setuptools.pypa.io/en/latest/userguide/development_mode.html
# https://packaging.python.org/en/latest/tutorials/packaging-projects/

import numpy as np
from util import generate_random_cluster_points

from fireferret.clustering.kmeans import KMeans

kmeans = KMeans(n_clusters=4)

c1 = generate_random_cluster_points(np.array([10, 5]), 50, 13)
c2 = generate_random_cluster_points(np.array([14, -5]), 50, 13)
c3 = generate_random_cluster_points(np.array([-6, -6]), 50, 8)
points = np.vstack([c1, c2, c3])

clusters, centroids = kmeans.fit(points, animate=True)

