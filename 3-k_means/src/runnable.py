import numpy as np
import sys

from sklearn.cluster import MiniBatchKMeans

N_CENTERS = 200
# The number of times the k-means algorithms passes over the whole dataset at hand.
N_PASSES = 1

def k_means(data, n_iterations):
    # Center initialization: 200 random points from dataset.
    np.random.shuffle(data)
    centers = data[:N_CENTERS,:]
    np.random.shuffle(data)

    # n_updates keeps track of how frequently each center has been updated.
    n_updates = np.zeros(N_CENTERS)
    # Online k-means algorithm - stochastic gradient descent.
    for iteration in range(n_iterations):
        x = data[iteration % data.shape[0]]
        min_distance = sys.maxint
        # For a given datapoint, find the center that is closest to it.
        for center_index in range(N_CENTERS):
            distance = np.linalg.norm(x - centers[center_index])
            if distance < min_distance:
                min_distance = distance
                closest_center_index = center_index
        n_updates[closest_center_index] += 1
        # Use a learning rate of 1 / #updates(center).
        centers[closest_center_index] +=  ((x - centers[closest_center_index])
                * 2 / n_updates[closest_center_index])
    return centers

def mapper(key, value):
    # TODO(f-daniel): checkout k-means++ for center updates
    yield "key", k_means(value, N_PASSES * value.shape[0])
    # kmeans = MiniBatchKMeans(init='k-means++', n_clusters=200)
    # kmeans.fit(value)
    # centroids = kmeans.cluster_centers_
    # yield "key", centroids

def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, values) pair here.
    yield k_means(values, N_PASSES * values.shape[0])
    # kmeans = MiniBatchKMeans(init='k-means++', n_clusters=200)
    # kmeans.fit(values)
    # centroids = kmeans.cluster_centers_
    # yield centroids
