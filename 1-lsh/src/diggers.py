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

def create_signature(hash_functions, shingles):
    signature = np.ones(len(hash_functions)) * sys.maxint
    for j in range(len(hash_functions)):
        for i in range(len(shingles)):
            signature[j] = np.minimum(hash_functions[j](shingles[i]), signature[j])
    return signature

def mapper(key, value):
    # key: None
    # value: one line of input file
    # initialize int-document array and id
    shingles = value.split(' ')
    document_id = int(shingles[0].split('_')[1])
    shingles = map(int, shingles[1:])
    n_rows = N_BANDS * N_ROWS_PER_BAND
    np.random.seed(seed=SEED)

    hash_functions = map(lambda x: gen_hashfunc(n_rows), range(n_rows))

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

# Simple test case for min hashing. Inspired by lecture slide 18 from week 3.
def test_min_hash():
    shingles_1 = [0, 1, 5, 6]
    shingles_2 = [2, 3, 4]
    shingles_3 = [0, 5, 6]
    permutation_1 = [2, 3, 6, 5, 0, 1, 4]
    permutation_2 = [3, 1, 0, 2, 5, 6, 4]
    permutation_3 = [0, 2, 6, 5, 1, 4, 3]
    # Create mock hash functions permuting the shingles.
    hash_functions = []
    hash_functions.append(lambda s: permutation_1[s % 7])
    hash_functions.append(lambda s: permutation_2[s % 7])
    hash_functions.append(lambda s: permutation_3[s % 7])
    signature_1 = create_signature(hash_functions, shingles_1)
    signature_2 = create_signature(hash_functions, shingles_2)
    signature_3 = create_signature(hash_functions, shingles_3)
    print signature_1
    print signature_2
    print signature_3

def min_hash():
    return 0

def band_hash():
    return 0
