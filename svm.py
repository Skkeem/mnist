import numpy as np
from numpy import linalg

""" In order to solve the Lagrangian dual problem,
I use the well known optimization solver library, cvxopt(http://cvxopt.org)"""
import cvxopt
import cvxopt.solvers


import theano
import theano.tensor as T
import input_data


def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x1, x2, p=4):
    return (1 + np.dot(x1, x2)) ** p

def gaussian_kernel(x1, x2, sigma=5.0):
    return np.exp(-linalg.norm(x1 - x2) ** 2 / (2 * (sigma ** 2)))


class SVM(object):

    def __init__(self, kernel=linear_kernel, C=None):
        self.kernel = kernel
        self.C = C
        if self.C is not None:
            self.C = float(self.C)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # print y

        # map the features into the other hyperplane with kernel function
        K = np.zeros( (n_samples, n_samples) )
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.kernel(X[i], X[j])

        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples))
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            t1 = np.diag(np.ones(n_samples) * -1)
            t2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack( (t1, t2) ))
            t1 = np.zeros(n_samples)
            t2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack( (t1, t2) ))

        # solve the QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # get Lagrange multipliers
        a = np.ravel(solution['x'])

        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        print "%d support vectors out of %d points" % (len(self.a), n_samples)

        # calculate b
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n], sv])
        if len(self.a) > 0:
            self.b /= len(self.a)

        # calculate w
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            # mark as "kernel mapping is necessary"
            self.w = None

    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            # apply the kernel function
            y_pred = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel(X[i], sv)
                y_pred[i] = s
            return y_pred + self.b

    def predict(self, X):
        return np.sign(self.project(X))


"""
The classifier should classify a number between 0 to 9.
Since SVM is basicallly a binary classifier,
10 SVM's are internally maintained,
y is predicted as the one with the largest positive margin.
"""
class DigitSVM(object):

    def __init__(self, kernel=linear_kernel, C=None):
        self.kernel = kernel
        self.C = C
        if self.C is not None:
            self.C = float(self.C)

    def fit(self, X, y):
        SVMs = []
        for i in range(0, 10):
            print "...Training for number %d" % ( i )
            svm = SVM()
            #svm.fit(X, map(lambda l: np.float64(1) if l == i else np.float64(-1), y))
            svm.fit(X, np.array(map(lambda l: 1. if l == i else -1., y)))
            SVMs.append( svm )

        self.SVMs = SVMs


    def predict(self, X):
        # execute 10 classifiers and choose the largest margin
        max_margin = np.zeros( X.shape[0] )
        num = np.zeros( X.shape[0], dtype=np.int8 )
        for i in range(0, 10):
            margin = self.SVMs[i].project(X)
            for j in range(len(margin)):
                if margin[j] > max_margin[j]:
                    max_margin[j] = margin[j]
                    num[j] = i

        return num



def load_data():
    mnist = input_data.read_data_sets(".")

    print 'Reading data set complete'

    return mnist


if __name__ == "__main__":


    datasets = load_data()

    #print datasets.validation.images.shape
    #print datasets.validation.labels.shape
    #print datasets.validation.labels

    SIZE = 5000
    train_images = datasets.train.images[:SIZE]
    train_labels = datasets.train.labels[:SIZE]

    test_images = datasets.test.images[:1000]
    test_labels = datasets.test.labels[:1000]


    digitSVM = DigitSVM()
    digitSVM.fit(train_images, train_labels)

    y_predict = digitSVM.predict(test_images)
    correct = np.sum(y_predict == test_labels)
    print "%d out of %d predictions correct" % (correct, len(y_predict))

