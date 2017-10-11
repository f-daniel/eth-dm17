import sys
import numpy as np

# The number of bands in signature matrix.
N_BANDS = 20
# The number of rows per band.
N_ROWS_PER_BAND = 50
# A large prime larger than the number of rows in the signature matrix, used for hashing.
PRIME = 3373
# Seed used for randomization in every instance of mapper.
SEED = 1337

def mapper(key, value):
    # key: None
    # value: one line of input file
    # initialize int-document array and id
    shingles = value.split(' ')
    document_id = int(shingles[0].split('_')[1])
    shingles = map(int, shingles[1:])
    n_rows = N_BANDS * N_ROWS_PER_BAND
    np.random.seed(seed=SEED)
    sigvec = np.ones(n_rows) * sys.maxint

    hash_functions = map(lambda x: gen_hashfunc(n_rows), range(n_rows))

    for j in range(n_rows):
        for i in range(len(shingles)):
            sigvec[j] = np.minimum(hash_functions[j](shingles[i]), sigvec[j])
    print(sigvec)
    if False:
        yield "key", "value"  # this is how you yield a key, value pair

def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    if False:
        yield "key", "value"  # this is how you yield a key, value pair

def gen_hashfunc(n_rows):
    a = np.random.randint(1, n_rows)
    b = np.random.randint(0, n_rows)
    return lambda s: ((np.multiply(a, s) + b) % PRIME) % n_rows

def min_hash():
    return 0

def band_hash():
    return 0
