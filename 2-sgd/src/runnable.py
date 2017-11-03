import numpy as np

def transform(X):
    # Make sure this function works for both 1D and 2D NumPy arrays.
    return X


def mapper(key, value):
    # key: None
    # value: one line of input file
    for image_data in value:
        image_data = map(float, image_data.split(" "))
        y = np.array(image_data[0])
        x = np.array(image_data[1:])
    # The yielded key should always be the same.
    yield "key", "value"  # This is how you yield a key, value pair

def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    yield np.random.randn(400)
