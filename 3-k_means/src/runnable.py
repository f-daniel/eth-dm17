import numpy as np
import sys
import sklearn.cluster
from sklearn.cluster.k_means_ import _k_init

N_CENTERS = 200
N_DIM = 250
# The number of times the k-means algorithms passes over the whole dataset at hand.
N_PASSES = 1

def k_means_online(data, n_iterations):
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

def k_means(data):
    np.random.shuffle(data)
    centers = data[:N_CENTERS,:]
    np.random.shuffle(data)

    #associated_centers = np.zeros(data.shape(0), 1)
    associated_dict = {k : [] for k in range(N_CENTERS)}
    iteration = 0
    change_in_centers = sys.maxint
    while change_in_centers > 1e-2 and iteration < 10:
        for point_index in range(data.shape[0]):
            x = data[point_index]
            min_distance = sys.maxint
            # For a given datapoint, find the center that is closest to it.
            for center_index in range(N_CENTERS):
                distance = np.linalg.norm(x - centers[center_index])
                if distance < min_distance:
                    min_distance = distance
                    closest_center_index = center_index
            # n_updates[closest_center_index] += 1
            #associated_centers[point_index] = closest_center_index
            associated_dict[closest_center_index].append(point_index)

        # all points assigned a closest center - recalculate each center as mean of 
        # points associated with it
        prev_centers = np.copy(centers)
        for c_i, c in enumerate(centers):
            print(associated_dict[c_i])
            c = np.mean([data[pt, :] for pt in associated_dict[c_i]], axis = 0)
            # print("c.shape: ", c.shape)
        print("centers.shape", centers.shape, "prec_centers.shape: ", prev_centers.shape)
        change_in_centers = np.linalg.norm(centers - prev_centers)
        print(iteration, ": ", change_in_centers)
    
    # for i, pt_i in enumerate(data):
    #   min_distance = sys.maxint
    #   for j, c_j in enumerate(centers):
    #       distance = np.linalg.norm(c_j - pt_i)
    #       if distance < min_distance:
    #           min_distance = distance
    return centers

def assign_labels_array(data, data_squared_norms, distances, centers):
    n_points = data.shape[0]
    closest_center = np.zeros(data.shape[0], dtype=np.int64)  # TODO: better to use float64?
    center_squared_norms = np.zeros(N_CENTERS, dtype=np.float32)
    inertia = 0.0
    # data_stride = data.strides[1] / sizeof(float)
    # center_stride = centers.strides[1] / sizeof(float)
    # dot = sdot # define dot product: np.dot ?

    if n_points == distances.shape[0]:
        store_distances = 1

    for center_index in range(N_CENTERS):
        center_squared_norms[center_index] = np.dot(centers[center_index, :], centers[center_index, :])

    for data_index in range(n_points):
        print("data_index: ", data_index)
        min_distance = -1
        for center_index in range(N_CENTERS):
            dist = 0.0
            # hardcoded: minimize euclidean distance to cluster center:
            # ||a - b||^2 = ||a||^2 + ||b||^2 -2 <a, b>
            dist += np.dot(data[data_index, :], centers[center_index, :])
            dist *= -2
            dist += center_squared_norms[center_index]
            dist += data_squared_norms[data_index]
            if min_distance == -1 or dist < min_distance:
                min_distance = dist
                closest_center[data_index] = center_index

        
        distances[data_index] = min_distance
        inertia += min_distance

    return inertia, closest_center, distances

def get_centers(data, closest_centers, distances):
    n_points = data.shape[0]
    n_centers = N_CENTERS
    centers = np.zeros((n_centers, N_DIM))
    print(closest_centers[:100])
    print("closest_centers shape: ", closest_centers.shape)
    n_points_per_center = np.bincount(closest_centers, minlength = n_centers)
    empty_clusters = np.where(n_points_per_center == 0)[0]

    if len(empty_clusters):
        # find points to reassign empty clusters to
        far_from_centers = distances.argsort()[::-1]

    for i, cluster_id in enumerate(empty_clusters):
        # XXX two relocated clusters could be close to each other
        new_center = data[far_from_centers[i]]
        centers[cluster_id] = new_center
        n_points_per_center[cluster_id] = 1

    for i in range(n_points):
        for j in range(N_DIM):
            centers[closest_centers[i], j] += data[i, j]

    centers /= n_points_per_center[:, np.newaxis]
    return centers

def compute_centers_and_inertia(data, data_squared_norms, centers):
    cluster_centers = -np.ones(N_CENTERS, np.int32)
    distances = np.zeros(shape=(data.shape[0], 1), dtype=data.dtype)
    inertia, cluster_centers, distances = assign_labels_array(data, data_squared_norms, distances, centers)
    return inertia, cluster_centers, distances

def k_means_sk(data):

    eps = 1e-4
    n_iter = 3 # 300
    n_points = data.shape[0]

    best_labels, best_inertia, best_centers = None, None, None
    # add tolerance() function which is dependent on variance
    
    # add precompute_distances?
    
    # subtract mean from input
    data_mean = np.mean(data, axis = 0)
    data -= data_mean

    # precompute squared normas of each data point (i.e. rows?)?
    data_squared_norms = np.sqrt((data * data).sum(axis=1))
    # add 'elkan' implementation?
    
    n_restarts = 3
    closest_centers = None
    for run in range(n_restarts):
        # TODO: initialize centers using k-means++
        centers = np.random.permutation(data)[:N_CENTERS]
        for i in range(n_iter):
            centers_old = centers.copy()
            print("centers.shape: ", centers.shape)
            # labels, intertia (E)
            # cluster_centers = -np.ones(N_CENTERS, np.int32)
            # distances = np.zeros(shape=(data.shape[0], 1), dtype=data.dtype)
            # inertia, cluster_centers, distances = assign_labels_array(data, data_squared_norms, distances, centers)
            inertia, cluster_centers, distances = compute_centers_and_inertia(data, data_squared_norms, centers)
            
            # means (M)
            centers = get_centers(data, cluster_centers, distances)
            print("centers.shape after get_centers: ", centers.shape)
            # check if new centers intertia etc. are better and if so, reassign
            if best_inertia is None or inertia < best_inertia:
                best_labels = cluster_centers.copy()
                best_centers = centers.copy()
                best_inertia = inertia

            # calculate change in centers - check if less than tolerance. If so, 
            # end of loop, break.
            centers_diff = (centers_old - centers)
            center_shift_total = np.dot(np.ravel(centers_diff, order='K'), 
                np.ravel(centers_diff, order='K'))
            if center_shift_total <= eps:
                if verbose:
                    print("Converged at iteration %d: "
                          "center shift %e within tolerance %e"
                          % (i, center_shift_total, eps))
                break
            
        # rerun E-step if not converged
        if center_shift_total > 0:
            # # rerun E-step in case of non-convergence so that predicted labels
            # match cluster centers
            best_inertia, best_labels, _ = \
                compute_centers_and_inertia(data, data_squared_norms, best_centers)
            print("Rerunning labels_inertia")
        
        # check in outer loop if returned centers are better than from previous
        # iterations and if so, reassign.   
        # labels, intertia, centers, n_iter_ = kmeans_single(data)
        if best_inertia is None or inertia < best_inertia:
                best_labels = labels.copy()
                best_centers = centers.copy()
                best_inertia = inertia
    # add mean to best centers again
    print("adding mean")
    best_centers += data_mean
    return best_centers, best_labels, best_inertia

def mapper(key, value):
    # TODO(f-daniel): checkout k-means++ for center updates
    # centers = _k_init(value, 200, np.linalg.norm(value, axis = 1), random_state = np.random.RandomState(seed = None))
    # yield "key", centers
    np.random.shuffle(value)
    yield "key", value#[:200,:]#k_means_online(value, N_PASSES * value.shape[0])

def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, values) pair here.
    results = k_means_sk(values)
    result_centers = results[0]
    yield result_centers
    # result = sklearn.cluster.KMeans(n_clusters = 200, verbose = 1).fit(values)
    # yield result.cluster_centers_
    # yield k_means(values)
    # yield k_means_online(values, N_PASSES * values.shape[0])
    #
