from __future__ import division
import numpy as np
import os
import psutil
import random

N_FEATURES = 250
N_DIM = int(N_FEATURES + (N_FEATURES  * (N_FEATURES + 1)) / 2)
N_DATA = 2000

def random_quadratic_transform(x):
    # Include dimension 1 features.
    ## OPTION1: Random feature subsets.
    #random.seed(2)
    ## Choose N_FEATURES many random feature indexes.
    #feature_indexes = map(lambda x: random.randint(0, 399), range(N_FEATURES))
    #features = np.take(x, feature_indexes)
    ## OPTION2: Use full input feature vector.
    features = x

    # Include squares of each dimension.
    features = np.concatenate((features, features * features))
    # Include sqrt(2)*x_i*x_j for i != j.
    for i in range(N_FEATURES-1):
        features = np.concatenate((features, np.sqrt(2) * features[i] * features[i+1:N_FEATURES]), axis = 0)
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0]/2.**30
    if memoryUse > 0.8:
        print('memory use:', memoryUse)
    return features

def transform(x):
    # 2D transformation.
    if x.ndim > 1:
        x_t = np.apply_along_axis(random_quadratic_transform, 1, x)
    # 1D transformation.
    else:
        # print '1d'
        x_t = random_quadratic_transform(x)
    return x_t

def mapper(key, value):
    # key: None
    # value: one line of input file
    first_moment_decay  = 0.9
    second_moment_decay = 0.999
    first_moment = np.zeros(N_DIM)
    second_moment = np.zeros(N_DIM)
    learning_rate = 0.001
    weight = np.random.rand(N_DIM)
    # weight = np.zeros(N_DIM)
    # Normalize weight's norm to be smaller or equal 1.
    weight = weight / np.linalg.norm(weight)
    convergence_threshold = 0.000001
    relative_change = 1
    t = 0
    epsilon = 1e-8
    n_iterations = 16000
    np.random.shuffle(value)
    # Convert sample strings to lists of floats.
    value = map(lambda sample: map(float, sample.split(" ")), value)
    ys = np.array([value[i][0] for i in range(N_DATA)])
    xs = np.array([value[i][1:] for i in range(N_DATA)])
    xs = transform(xs)
    # Adam.
    while relative_change >= convergence_threshold and t < n_iterations:
        y = ys[t % N_DATA]
        # x = transform(xs[t % N_DATA])
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
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
    print('memory use:', memoryUse)
    print t
    yield "key", np.transpose(weight)  # This is how you yield a key, value pair

def reducer(key, values):
    weight = np.divide(np.sum(values, axis = 0), len(values))
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    yield weight
