
import pickle
import six
import numpy as np
import argparse


def read_pkl(fname):
    with open(fname, 'rb') as d_pickle:
        data = six.moves.cPickle.load(d_pickle)
    return data


class Perceptron:
    """Train and test the perceptron.
    This class has one perceptron model, which has weight and bias.
    Before setting the class, input "Train data" and "Test data",
    in order to set feature matrix due to feature dimension.

    """
    def __init__(self, traindata, testdata):
        self.num_ofdim = np.max([len(traindata['x'][0]), len(testdata['x'][0])])
        self.num_ofclass = int(np.max(traindata['y']) + 1)
        self.w = np.random.rand(self.num_ofclass, self.num_ofdim + 1)

    def reshape_mat(self, data):
        """Reshaping input feature matrix, because either train/test data
        does not always have all the features that the other has.
        """
        if len(data['x'][0]) < self.num_ofdim:
            x = data['x']
            x_plus = np.zeros([len(x), self.num_ofdim - len(x[0])])
            data['x'] = np.append(x, x_plus, axis=1)
        return data

    def predict(self, w, x):
        """Prediction function.
        N-class discrimination: from 0 to (n-1)
        """
        pred_val = np.dot(w, np.append(x, 1))
        pred_class = np.argmax(pred_val)
        return pred_class

    def w_update(self, x, y, pred_class, alpha=0.001):
        """Weight update function.
        alpha is learning rate.
        """
        tempw = self.w[pred_class]
        self.w[y] += alpha * np.append(x, 1)
        self.w[pred_class] -= alpha * np.append(x, 1)

    def train(self, data, iteration):
        """Train function.
        """
        num = len(data['y'])
        data = self.reshape_mat(data)

        for j in range(0, iteration):
            acc = 0
            order = np.random.permutation(num)
            for i in range(0, num):
                x, y = data['x'][order[i]], data['y'][order[i]]
                yhat = self.predict(self.w, x)
                if y == yhat:
                    acc += 1
                else:
                    self.w_update(x, y, yhat)
            accuracy = acc / float(num)
            print 'Iteration %d: acc = %f' % (j+1, accuracy*100)

    def test(self, data):
        """Test function.
        """
        num = len(data['y'])
        self.reshape_mat(data)

        acc = 0
        order = np.random.permutation(num)
        for i in range(0, num):
            x, y = data['x'][order[i]], data['y'][order[i]]
            yhat = self.predict(self.w, x)
            if y == yhat:
                acc += 1
        accuracy = acc / float(num)
        print 'Test: acc = %f' % (accuracy * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read data file')
    fname = "mnist"

    traindata = read_pkl("data/" + fname + ".pkl")
    testdata = read_pkl("data/" + fname + ".t.pkl")

    Perc = Perceptron(traindata, testdata)

#    Perc.test(traindata)
    Perc.train(traindata, 10)
    Perc.test(testdata)

#    Perc.test(traindata)


"""
    def calc_acc(self,data):
        acc = 0
        num = self.traindatanum
        for i in range(0,num):
            x = data['x'][i]
            y = data['y'][i]
            yhat = self.predict(x,y)
            if y == yhat:
                acc += 1
        acc = acc / num
        return acc

"""
