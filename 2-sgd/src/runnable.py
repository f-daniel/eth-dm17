import numpy as np

def transform(X):
    # Make sure this function works for both 1D and 2D NumPy arrays.
    return X


def mapper(key, value):
    # key: None
    # value: one line of input file
    value = map(lambda x: map(float, x.split(" ")), value)
    # x has datapoints in rows and features in columns. Shape: (2000, 400).
    x = np.array(map(lambda x: x[1:], value))
    y = np.array(map(lambda x: x[0], value))
    # The yielded key should always be the same.
    #yield "key", "value"  # This is how you yield a key, value pair


def reducer(key, values):
    a = 5
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    #yield np.random.randn(400)
