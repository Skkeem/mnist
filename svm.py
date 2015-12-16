import numpy as np
from numpy import linalg

""" In order to solve the Lagrangian dual problem,
I use the well known optimization solver library, cvxopt(http://cvxopt.org)"""
import cvxopt
import cvxopt.solvers

import input_data


def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x1, x2, p=3):
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

    def margin(self, X):
        # get the distance to the separator
        if self.w is not None:
            return (np.dot(X, self.w) + self.b) / linalg.norm(self.w)
        else:
            y_pred = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel(X[i], sv)
                y_pred[i] = s

            # it's a bit difficult to calculate the precise value of ||w||.
            # instead use the similar equation as linear case
            weight = np.zeros(X.shape[1])
            for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                weight += a * sv_y * sv
            return (y_pred + self.b) / linalg.norm(weight)
                    

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
            svm = SVM(kernel=self.kernel, C=self.C)
            #svm.fit(X, map(lambda l: np.float64(1) if l == i else np.float64(-1), y))
            svm.fit(X, np.array(map(lambda l: 1. if l == i else -1., y)))
            SVMs.append( svm )

        self.SVMs = SVMs


    def predict(self, X):
        # execute 10 classifiers and choose the largest margin
        max_margin = np.zeros( X.shape[0] )
        num = np.zeros( X.shape[0], dtype=np.int8 )
        for i in range(0, 10):
            margin = self.SVMs[i].margin(X)
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

    """
    SIZE = 1000
    train_images = datasets.train.images[:SIZE]
    train_labels = datasets.train.labels[:SIZE]

    test_images = datasets.test.images[:1000]
    test_labels = datasets.test.labels[:1000]
    """

    train_images = datasets.train.images
    train_labels = datasets.train.labels

    test_images = datasets.test.images
    test_labels = datasets.test.labels

    digitSVM = DigitSVM(kernel=gaussian_kernel)
    digitSVM.fit(train_images, train_labels)

    y_predict = digitSVM.predict(test_images)
    correct = np.sum(y_predict == test_labels)
    print "%d out of %d predictions correct" % (correct, len(y_predict))


    """
    def test_linear():
        X1, y1, X2, y2 = gen_lin_separable_data()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        clf = SVM()
        clf.fit(X_train, y_train)

        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print "%d out of %d predictions correct" % (correct, len(y_predict))

    def test_non_linear():
        X1, y1, X2, y2 = gen_non_lin_separable_data()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        clf = SVM(gaussian_kernel)
        clf.fit(X_train, y_train)

        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print "%d out of %d predictions correct" % (correct, len(y_predict))

    def test_soft():
        X1, y1, X2, y2 = gen_lin_separable_overlap_data()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        clf = SVM(C=0.1)
        clf.fit(X_train, y_train)

        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print "%d out of %d predictions correct" % (correct, len(y_predict))


    def gen_lin_separable_data():
        # generate training data in the 2-d case
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[0.8, 0.6], [0.6, 0.8]])
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def gen_non_lin_separable_data():
        mean1 = [-1, 2]
        mean2 = [1, -1]
        mean3 = [4, -4]
        mean4 = [-4, 4]
        cov = [[1.0,0.8], [0.8, 1.0]]
        X1 = np.random.multivariate_normal(mean1, cov, 50)
        X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, 50)))
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 50)
        X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, 50)))
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def gen_lin_separable_overlap_data():
        # generate training data in the 2-d case
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[1.5, 1.0], [1.0, 1.5]])
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def split_train(X1, y1, X2, y2):
        X1_train = X1[:90]
        y1_train = y1[:90]
        X2_train = X2[:90]
        y2_train = y2[:90]
        X_train = np.vstack((X1_train, X2_train))
        y_train = np.hstack((y1_train, y2_train))
        return X_train, y_train

    def split_test(X1, y1, X2, y2):
        X1_test = X1[90:]
        y1_test = y1[90:]
        X2_test = X2[90:]
        y2_test = y2[90:]
        X_test = np.vstack((X1_test, X2_test))
        y_test = np.hstack((y1_test, y2_test))
        return X_test, y_test


    print "test linear"
    test_linear()

    print "test linear"
    test_linear()

    print "test non linear"
    test_non_linear()
    """
