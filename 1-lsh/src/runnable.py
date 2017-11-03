from __future__ import division
import sys
import numpy as np


# The number of bands in signature matrix.
N_BANDS = 100
# The number of rows per band.
N_ROWS_PER_BAND = 10
# Number of buckets for band hashing.
N_BUCKETS = 8193
# A large prime larger than the number of rows in the signature matrix, used for hashing.
PRIME = 3373
# Seed used for randomization in every instance of mapper.
SEED = 1337
# Similarity required for a document pair to be emitted.
SIMILARITY = .85


def create_signature(hash_params, shingles):
    """Creates the signature of a document by applying min hashing based on the
    hash_params to the shingles column. A hash function is of the form:
    h(s) = ((a * s  + b ) mod PRIME) mod n_rows."""
    n_rows = len(hash_params[0])
    signature = np.ones(n_rows) * sys.maxint
    param_as, param_bs = hash_params
    shingles = np.resize(shingles, (1, len(shingles)))
    param_as = np.resize(param_as, (n_rows, 1))
    param_bs = np.resize(param_bs, (n_rows, 1))
    # This matrix is neither the shingles nor the signature matrix. It is a matrix, with rows
    # representing hash functions and columns representing single shingles of the same
    # document.
    matrix = np.multiply(param_as, shingles)
    matrix = np.add(matrix, param_bs)
    matrix = np.mod(matrix, PRIME)
    matrix = np.mod(matrix, n_rows)
    signature = np.amin(matrix, 1)
    return signature

def similarity(shingles_1, shingles_2):
    """Computes the Jaccard similarity of two lists of shingles."""
    set_a = set(shingles_1)
    set_b = set(shingles_2)
    intersection_length = len(set_a.intersection(set_b))
    return intersection_length / (len(set_a) + len(set_b) - intersection_length)

def mapper(key, value):
    """The mapper's key is None. Its value is an input line. Its output keys are string
    concatenations of band id and bucket its. Its output values are document id and
    shingles that have been mapped into the named into the referred bucket for the
    referred band."""
    shingles = value.split(' ')
    document_id = int(shingles[0].split('_')[1])
    shingles = np.array(map(int, shingles[1:]))
    n_rows = N_BANDS * N_ROWS_PER_BAND
    np.random.seed(seed=SEED)

    hash_params = gen_hash_params(n_rows, n_rows)
    signature = create_signature(hash_params, shingles)

    param_as, param_bs = gen_hash_params(N_BUCKETS, N_ROWS_PER_BAND)
    for i in range(N_BANDS):
        band = np.copy(signature[i * N_ROWS_PER_BAND : (i + 1) * N_ROWS_PER_BAND])
        band = np.mod(np.mod(np.multiply(band, param_as) + param_bs, PRIME),
                      N_BUCKETS)
        # Arbitraty format combining both band id as well as the hashed-into bucket.
        key = str(i) + '-' + str(np.sum(band) % N_BUCKETS)
        yield key, [document_id, shingles]

def reducer(key, values):
    """See mapper function for reducer input. The reducer's output are similar document
    pairs."""
    if len(values) < 2:
        return
    for i in range(len(values)):
        [document_id_1, shingles_1] = values[i]
        for j in range(i + 1, len(values)):
            [document_id_2, shingles_2] = values[j]
            if similarity(shingles_1, shingles_2) >= SIMILARITY:
                yield min(document_id_1, document_id_2), max(document_id_1, document_id_2)

def gen_hash_params(n_buckets, n_hash_functions):
    """Returns a list n_hash_functions many a parameters and a list of b parameters. These
    parameters are used for hashing. The as and bs are a function of n_buckets."""
    param_as = np.random.randint(1, n_buckets, n_hash_functions)
    param_bs = np.random.randint(0, n_buckets, n_hash_functions)
    return (param_as, param_bs)
