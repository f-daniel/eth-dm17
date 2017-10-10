import numpy as np
import sys

def mapper(key, value):
    # key: None
    # value: one line of input file

    # initialize int-document array and id
    # r - number of rows per band
    # b - number of bands
    shingles = value.split(' ')
    id = int (shingles[0].split('_')[1])
    shingles = map(int, shingles[1:])
    r = 50  
    b = 20
    nrows = r*b
    np.random.seed(seed=1337)
    sigvec = np.ones(nrows)*sys.maxint

    # kevins fault
    hf_vec = map(gen_hashfunc(nrows), range(nrows))

    for i in range(len(shingles)):
    	if shingles[i] != 0:
    		for j in range(nrows):
    			sigvec[i] = np.minimum(hf_vec[j](shingles[i]), sigvec[i])

    #print(sigvec)

    if False:
        yield "key", "value"  # this is how you yield a key, value pair


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    if False:
        yield "key", "value"  # this is how you yield a key, value pair


def gen_hashfunc(nrows):
	a = np.random.randint(1,nrows)
	b = np.random.randint(0,nrows)
	p = 3373 # large prime > nrows
	return lambda s:((np.multiply(a,s)+b)%p)%nrows

def min_hash():
	return 0 

def band_hash():
	return 0

