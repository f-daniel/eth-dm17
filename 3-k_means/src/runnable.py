from __future__ import division

import numpy as np
from numpy.random import randint, choice, shuffle
from numpy.linalg import norm
import sys
import coresets

N_CLUSTERS = 200
N_ITERATIONS = 10
N_SAMPLES = 10000

def mapper(key, value):
    yield "key", value

def reducer(key, values):
    # Size of the dataset.
    n = values.shape[0]
    # Dimensionality of the data.
    d = values.shape[1]

    # Initialize cluster centers with D^2 sampling. This is the same
    # initialization that is used in k-means++.

    # Array containing the distances from each center to each datapoint.
    distances = np.full((N_CLUSTERS - 1, n), sys.maxint)
    centers = np.empty((N_CLUSTERS, d))
    center = values[randint(n)]
    centers[0] = center
    for cluster_index in range(1, N_CLUSTERS):
        distances[cluster_index - 1] = norm(values - center, axis = 1)**2
        min_distances = distances.min(axis = 0)
        p = min_distances / min_distances.sum()
        center = values[choice(n, p = p)]
        centers[cluster_index] = center

    # Use a pseudo coreset begin a uniformly sampled subset of the data with
    # equal weights (1).
    np.random.shuffle(values)
    # coreset_var = [(values[i], 1) for i in range(N_SAMPLES)]
    coreset_var = coresets.get_coreset(values, N_CLUSTERS, coreset_size = 10000, delta=0.1)
    for iteration in range(N_ITERATIONS):
        # The sum of points that are closest to a given cluster center.
        cluster_sum = np.zeros((N_CLUSTERS, d))
        # The number of points that are closest to a given cluster center.
        cluster_count = np.zeros(N_CLUSTERS)
        for point, w in coreset_var:
            closest_cluster_index = norm(centers - point, axis = 1).argmin(axis
                    = 0)
            cluster_sum[closest_cluster_index] += w * point
            cluster_count[closest_cluster_index] += w
        for cluster_index in range(N_CLUSTERS):
            if cluster_count[cluster_index] > 0:
                centers[cluster_index] = (cluster_sum[cluster_index] /
                        cluster_count[cluster_index])
    yield centers
