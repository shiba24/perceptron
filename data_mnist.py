
# import pychecker.checker
import numpy as np
import argparse
import six
import pickle
from six.moves.urllib import request
import os
import bz2

trainurl = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.bz2'
testurl = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.t.bz2'

def read_file(filename):
    """Read data file and make features
    Args:
        filename: filename to read (str with relative path)

    Returns:
        ans: A dict mapping keys, consisting of:
            {'target': #TARGET, 'features': #np.2darray}
        max_dim: max dimension of input featureset

    Features 2darray has the shape of [[dimensions array],[valus array]].
    Be careful this is not direct expression of matrix.
    """
    print "reading and making pickles..."
    ans = []
    max_dim = 1
    for line in open(filename, 'r'):
        sind = line.find(" ")
        label = int(line[0:sind])    # separated by space
        one = {}
        one['target'] = label

        features = np.zeros([2, 1])
        length = line[sind + 1:].find(" ")
        while 1:
            tempstr = line[sind + 1:sind + 1 + length]
            if tempstr.find(':') > 0:
#                print tempstr
                fdim = int(tempstr[:tempstr.find(":")])
                # get max dimension of input feature to make matrix
                max_dim = np.max([max_dim, fdim])
                fval = float(tempstr[tempstr.find(":") + 1:])
                f = np.array([[fdim], [fval]])
                features = np.append(features, f, axis=1)
                sind, length = sind + length + 1, line[sind + length + 2:].find(" ")
            if length < 0:
                break

        features = features.T[1:].T
        one['features'] = features
        ans.append(one)
    return ans, max_dim


def make_featuremat(dicts, dim):
    """Make feature matrix for input and output
    Args:
        dicts, dim: Returns of function "read_file"

    Returns:
        data: feature matrix
    """
    data = {}
    data['x'] = np.zeros([len(dicts), dim])
    data['y'] = np.zeros([len(dicts)])
    for i in range(0, len(dicts)):
        data['y'][i] = dicts[i]['target']
        temp = dicts[i]['features']

        for j in range(0, len(temp[0])):
            data['x'][i][temp[0][j] - 1] = temp[1][j]

    return data


def save_pkl(data, filename):
    savename = filename + '.pkl'
    print 'Save datasets as pkl: %s' % (savename)
    with open(savename, 'wb') as output:
        six.moves.cPickle.dump(data, output, -1)
    print('Done')


def download_dataset():
    print 'Downloading data/mnist.bz2'
    request.urlretrieve(trainurl, filename="data/mnist.bz2")
    bz2.decompress("data/mnist.bz2")
    print('Done')
    print 'Downloading data/mnist.t.bz2'
    request.urlretrieve(testurl + '.t', filename="data/mnist.t.bz2")
    bz2.decompress("data/mnist.t.bz2")
    print('Done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read MNIST dataset')
    fname = "data/mnist"

    if not os.path.exists(fname):
        download_dataset()


    ans, max_dim = read_file(fname)
    print len(ans), max_dim
    data = make_featuremat(ans, max_dim)
    save_pkl(data, fname)


    fname = "data/mnist.t"
    ans, max_dim = read_file(fname)
    print len(ans), max_dim
    data = make_featuremat(ans, max_dim)
    save_pkl(data, fname)


"""
open pickle

with open(filename, 'rb') as D_pickle:
    D = six.moves.cPickle.load(D_pickle)
"""
