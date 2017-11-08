import numpy as np

# The regularization parameter for sgd, typically refered to as 'lambda'.
REGULARIZATION_PARAM = 0.5
# The number of iterations for sgd.
N_ITERATIONS = 1000
DIMENSIONALITY = 400

def transform(X):
    # Make sure this function works for both 1D and 2D NumPy arrays.
    return X


def mapper(key, value):
    # key: None
    # value: one line of input file
    first_moment_decay  = 0.9
    second_moment_decay = 0.999
    bias_previous = np.array([0])
    bias = np.array([0.1])
    first_moment = np.zeros(DIMENSIONALITY + 1)
    second_moment = np.zeros(DIMENSIONALITY + 1)
    learning_rate = 0.001
    weight_previous = np.zeros(DIMENSIONALITY)
    weight = np.array(map(float, value[0].split(" ")[1:]))
    convergence_threshold = 0.01
    relative_change = 1
    t = 0
    epsilon = 1e-8
    while relative_change >= convergence_threshold and t < 2000:
        image_data = map(float, value[t].split(" "))
        y = np.array(image_data[0])
        x = np.array(image_data[1:])
        t = t + 1
        margin_previous = np.multiply(y, np.multiply(weight_previous, x) + bias)
        margin = np.multiply(y, np.multiply(weight, x) + bias)
        gradient = np.divide(np.maximum(np.zeros(DIMENSIONALITY), np.ones(DIMENSIONALITY) - margin) - np.maximum(np.zeros(DIMENSIONALITY), np.ones(DIMENSIONALITY) - margin_previous),
                np.absolute(weight - weight_previous))
        print np.absolute(bias - bias_previous).shape
        print np.array([(max(0, 1 - y * (np.dot(weight, x) + bias)) - max(0, 1 - y * (np.dot(weight, x) + bias_previous))) / np.absolute(bias - bias_previous)])[:,0].shape
        gradient = np.concatenate((gradient, np.array([(max(0, 1 - y * (np.dot(weight, x) + bias)) - max(0, 1 - y * (np.dot(weight, x) + bias_previous))) / np.absolute(bias - bias_previous)])[:,0]))
        first_moment = first_moment * first_moment_decay + (1 - first_moment_decay) * gradient
        second_moment = second_moment * second_moment_decay + (1 - second_moment_decay) * gradient**2
        corrected_first_moment = first_moment / (1 - first_moment_decay**t)
        corrected_second_moment = second_moment / (1 - second_moment_decay**t)
        weight_previous = weight
        bias_previous = bias
        weight = weight - np.divide(learning_rate * corrected_first_moment[0:400], np.sqrt(corrected_second_moment[0:400]) + epsilon)
        bias = bias - learning_rate * corrected_first_moment[400] / np.sqrt(corrected_second_moment[400] + epsilon)
        aux = np.concatenate((weight, bias))
        relative_change = np.linalg.norm(aux - np.concatenate((weight_previous, bias_previous))) / np.linalg.norm(aux)
    # The yielded key should always be the same.
    yield "key", np.transpose(weight)  # This is how you yield a key, value pair

def reducer(key, values):
    weight = np.divide(np.sum(values, axis = 0), len(values))
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    yield weight
