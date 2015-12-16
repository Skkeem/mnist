import os
import sys
import timeit
import matplotlib.pyplot as plt

import numpy as np

import theano
import theano.tensor as T

from logistic_regression import LogisticRegression, load_data

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
            activation=T.tanh):
        '''
        Hidden layer of MLP: units are fully-connected and have sigmoidal
        activation function. Weight matrix W is of shape(n_in, n_out) and the
        bias vector b is of shape (n_out,).

        @rng
        -type : numpy.random.RandomState
        -param : a random number generator used to initialize weights

        @input
        -type : theano.tensor.dmatrix
        -param : a symbolic tensor of shape(n_examples, n_in)

        @n_in
        -type : int
        -param : dimensionality of input

        @n_out
        -type : int
        -param : number of hidden units

        @activation
        -type : theano.Op or function
        -param : Non linearity to be applied in the hidden layer
        '''
        self.input = input

        if W is None:
            W_values = np.asarray(
                    rng.uniform(
                        low=-np.sqrt(6. / (n_in + n_out)),
                        high=np.sqrt(6. / (n_in + n_out)),
                        size=(n_in, n_out)
                    ),
                    dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
            
            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
            
        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
                lin_output if activation is None
                else activation(lin_output)
        )
        self.params = [self.W, self.b]


class MLP(object):
    '''
    Multi-Layer Perceptron Class

    A multilayer perception is feedforward artificial neural network model that
    has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the sigmoid
    function while the top layer is a softmax layer.
    '''

    def __init__(self, rng, input, n_in, n_hidden, n_out):
        '''
        @rng
        -type : numpy.random.RandomState
        -param : a random number generator used to initialize weights

        @input
        -type : theano.tensor.TensorType
        -param : a symbolic variable that describes the input of the
        architecture

        @n_in
        -type : int
        -param : number of input units, the dimension of the space in which the
        datapoints lie

        @n_hidden
        -type : int
        -param : number of hidden units

        @n_out
        -type : int
        -param : number of output units, the dimension of the space in which
        the labels lie
        '''
        self.hiddenLayer = HiddenLayer(
                rng=rng,
                input=input,
                n_in=n_in,
                n_out=n_hidden,
                activation=T.tanh
        )
        self.logRegressionLayer = LogisticRegression(
                input=self.hiddenLayer.output,
                n_in=n_hidden,
                n_out=n_out
        )

        # Regularization options
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )
        self.L2_sqr = (
                (self.hiddenLayer.W ** 2).sum()
                + (self.logRegressionLayer.W ** 2).sum()
        )

        self.negative_log_likelihood = (
                self.logRegressionLayer.negative_log_likelihood
        )

        self.errors = self.logRegressionLayer.errors

        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

        self.input = input


def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
        batch_size=20, n_hidden=500):
    '''
    Stochastic gradient descent optimization for a multilayer perception

    @learning_rate
    -type : float
    -param : learning rate used

    @L1_reg
    -type : float
    -param : L1-norm's weight when added to the cost

    @L2_reg
    -type : float
    -param : L2-norm's weight when added to the cost

    @n_epochs
    -type : int
    -param : maximal number of epochs to run the optimizer
    '''
    datasets = load_data()
    
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    print '... building the model'

    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    rng = np.random.RandomState(1234)

    classifier = MLP(
            rng=rng,
            input=x,
            n_in=28 * 28,
            n_hidden=n_hidden,
            n_out=10
    )

    cost = (
            classifier.negative_log_likelihood(y)
            + L1_reg * classifier.L1
            + L2_reg * classifier.L2_sqr
    )

    test_model = theano.function(
            inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: test_set_x[index * batch_size:(index + 1) * batch_size],
                y: test_set_y[index * batch_size:(index + 1) * batch_size]
            }
    )

    in_sample_test_model = theano.function(
            inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size]
            }
    )

    validate_model = theano.function(
            inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size]
            }
    )

    gparams = [T.grad(cost, param) for param in classifier.params]

    updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(classifier.params, gparams)
    ]

    train_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size]
            }
    )

    print '... training'

    patience = 10000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False
    x_axis = []
    y_axis = []
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = [validate_model(i) for i
                        in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)

                print(
                        'epoch %i, minibatch %i/%i, validation error %f %%' %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            this_validation_loss * 100.
                        )
                )

                if this_validation_loss < best_validation_loss:
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    test_losses = [test_model(i) for i 
                            in xrange(n_test_batches)]
                    test_score = np.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                        'best model %f %%') %
                        (epoch, minibatch_index + 1, n_train_batches,
                         test_score * 100.))

                    x_axis.append(epoch)
                    y_axis.append(test_score * 100.)

                if patience <= iter:
                    done_looping = True
                    break

    in_sample_losses = [in_sample_test_model(i) for i
            in xrange(n_train_batches)]
    in_sample_score = np.mean(in_sample_losses)
    print('##in sample test error of %f %%' % (in_sample_score * 100.))

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
          'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
            os.path.split(__file__)[1] +
            ' ran for %.2fm' % ((end_time - start_time) / 60.))

    plt.plot(np.asarray(x_axis), np.asarray(y_axis))
    plt.show()


if __name__ == '__main__':
    test_mlp(n_epochs=500)
