from __future__ import division
import numpy as np

N_DIM = 400
N_DATA = 2000

def transform(x):
    x_transformed =  normalize(x)
    return x_transformed

def normalize(x):
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    return (x - x_mean) / x_std

def mapper(key, value):
    # key: None
    # value: one line of input file
    first_moment_decay  = 0.9
    second_moment_decay = 0.999
    first_moment = np.zeros(N_DIM)
    second_moment = np.zeros(N_DIM)
    learning_rate = 0.001
    weight = np.random.rand(N_DIM)
    # Normalize weight's norm to be smaller or equal 1.
    weight = weight / np.linalg.norm(weight)
    convergence_threshold = 0.000001
    relative_change = 1
    t = 0
    epsilon = 1e-8
    n_iterations = 100000
    np.random.shuffle(value)
    # Convert sample strings to lists of floats.
    value = map(lambda sample: map(float, sample.split(" ")), value)
    ys = np.array([value[i][0] for i in range(N_DATA)])
    xs = transform(np.array([value[i][1:] for i in range(N_DATA)]))
    # Adam.
    while t < n_iterations:
        y = ys[t % N_DATA]
        x = xs[t % N_DATA]
        t = t + 1
        # Analytic derivation via case distinction.
        if y * np.dot(weight, x) > 1:
            gradient = np.zeros(N_DIM)
        else:
            gradient = -y*x
        first_moment = first_moment * first_moment_decay + (1 - first_moment_decay) * gradient
        second_moment = (second_moment * second_moment_decay +
                (1 - second_moment_decay) * gradient**2)
        corrected_first_moment = first_moment / (1 - first_moment_decay**t)
        corrected_second_moment = second_moment / (1 - second_moment_decay**t)
        weight_previous = weight
        weight = weight - np.divide(learning_rate * corrected_first_moment,
                np.sqrt(corrected_second_moment) + epsilon)
        relative_change = np.linalg.norm(weight - weight_previous) / np.linalg.norm(weight_previous)
    # The yielded key should always be the same.
    print t
    yield "key", np.transpose(weight)  # This is how you yield a key, value pair

def reducer(key, values):
    weight = np.divide(np.sum(values, axis = 0), len(values))
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    yield weight
