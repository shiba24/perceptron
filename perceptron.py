
import pickle
import six
import numpy as np
import argparse

"""
TODO: Set Convergence condition
TODO: Alpha ?? 
TODO: What if there is not data?
"""


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
    def __init__(self,traindata,testdata):
        self.feature = np.max([len(traindata['x'][0]) , len(testdata['x'][0])])
        self.iniW = np.random.rand(self.feature)
        self.iniB = np.random.rand(1)


    def set_initial_wb(self):
        """Initial w & b will be set to be randomized, (mean , sd) = (0,1).
        In the future maybe need to be enable to set user's initial w & b.
        """
        self.w = np.append(self.iniW,self.iniB)


    def reshape_mat(self,data):
        """Reshaping input feature matrix, because either train/test data
        does not always have all the features that the other has.
        """
        if len(data['x'][0]) < self.feature:
            x = data['x']
            x_plus = np.zeros([len(x),self.feature - len(x[0])])
            data['x'] = np.append(x,x_plus,axis=1)
        return data


    def predict(self,w,x):
        """Prediction function.
        Two-class discrimination: +1 or -1
        """
        pred_val = np.dot(w,np.append(x,1))
        if pred_val >= 0:
            pred_class = 1
        else:
            pred_class = -1
        return pred_class


    def w_update(self,x,y,pred_class,alpha=0.0001):
        """Weight update function.
        alpha is learning rate.
        """
        w_new = self.w + alpha*(y - pred_class)*np.append(x,1)
        self.w = w_new


    def train(self,data,iteration):
        """Train function.
        Need to be fixed to calculate using matrix, not vector?
        """
        num = len(data['y'])
        self.set_initial_wb()
        self.reshape_mat(data)

        for j in range(0,iteration):
            acc = 0
            order = np.random.permutation(num)
            for i in range(0,num):
                x , y = data['x'][order[i]] , data['y'][order[i]]
                yhat = self.predict(self.w,x)
                if y == yhat:
                    acc += 1
                else:
                    self.w_update(x,y,yhat)
            accuracy = acc / float(num)
            print 'Iteration %d: acc = %f' % (j+1, accuracy*100)


    def test(self,data):
        """Test function.
        Need to be fixed to calculate using matrix, not vector?
        """
        num = len(data['y'])
        self.reshape_mat(data)

        acc = 0
        order = np.random.permutation(num)
        for i in range(0,num):
            x , y = data['x'][order[i]] , data['y'][order[i]]
            yhat = self.predict(self.w,x)
            if y == yhat:
                acc += 1
        accuracy = acc / float(num)        
        print 'Test: acc = %f' % (accuracy*100)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read data file')
    parser.add_argument('--filename', '-f', default="a1a", type=str, help='FILENAME')
    args = parser.parse_args()
    fname = args.filename

    traindata = read_pkl("data/"+fname+".pkl")
    testdata = read_pkl("data/"+fname+".t.pkl")

    Perc = Perceptron(traindata,testdata)

#    Perc.test(traindata)
    Perc.train(traindata,10)
    Perc.test(testdata)

    Perc.test(traindata)


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


