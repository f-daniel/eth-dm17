import numpy as np

def mapper(key, value):
    # key: None
    # value: one line of input file
    
    # TBD: checkout k-means++ for center updates

    np.random.shuffle(value)
    centers = value[:200,:]
    np.random.shuffle(value)
    update_count = np.zeros(200)

    # k-means algorithm
    for i in value:
        min_distance = 10e7
        for j_index, j in enumerate(centers):
            distance = np.linalg.norm(i-j)
            if distance < min_distance:
                min_distance = distance
                center_index = j_index
                update_count[j_index] += 1

        centers[center_index] += (2 * (i - centers[center_index]))/update_count[center_index]

    yield "key", centers  # this is how you yield a key, value pair


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    np.random.shuffle(values)
    centers = values[:200,:]
    np.random.shuffle(values)
    update_count = np.zeros(200)

    # k-means algorithm
    for i in values:
        min_distance = 10e7
        for j_index, j in enumerate(centers):
            distance = np.linalg.norm(i-j)
            if distance < min_distance:
                min_distance = distance
                center_index = j_index
                update_count[j_index] += 1

        centers[center_index] += (2 * (i - centers[center_index]))/update_count[center_index]



    # Note that we do *not* output a (key, values) pair here.
    yield centers
