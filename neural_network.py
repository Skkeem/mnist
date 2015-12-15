import os
import timeit
import cPickle
import sys

import numpy as np

import theano
import theano.tensor as T

import input_data

class NeuralNetwork(object):
    '''
    Multi-class Neural Network Class
    '''

    def __init__(self, input, n_in, n_out):
        '''
        Initialize the parameters of the neural network 

        @input
        -type : theano.tensor.TensorType
        -param : symbolic variable that describe the input of the architecture.
                 (one minibatch) 
        
        @n_in
        -type : int
        -param : number of input units, the dimension of the space in which
                 the datapoints lie.

        @n_out
        -type : int
        -param : number of output units, the dimension of the space in which
                 the labels lie.
        '''
        self.W = theano.shared(
                value=np.zeros((n_in, n_out),
                                dtype=theano.config.floatX
                ),
                name='W',
                borrow=True
        )

        self.b = theano.shared(
                value=np.zeros((n_out,),
                                dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
        )

        # predicted probability
        self.pred_prob = T.nnet.softmax(T.dot(input, self.W) + self.b) 

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.pred_prob, axis=1)

        self.params = [self.W, self.b]
        
        self.input = input

    def cross_entropy(self, y):
        '''
        Returns the mean of the cross-entropy of the prediction of this model
        under a given target distribution
        
        @y
        -type : theano.tensor.TensorType 
        -param : corresponds to a vector that gives for each example
                 the correct label (one-hot)
        '''
        return  -T.mean(T.log(self.pred_prob)[T.arange(y.shape[0]), y])
    
    def errors(self, y):
        '''
        Return a float representing the number of errors in the minibatch over
        the total number of examples of the minibatch ; zero one loss over
        the size of the minibatch
    
        @y
        -type : theano.tensor.TensorType 
        -param : corresponds to a vector that gives for each example
                 the correct label
        '''
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                    'y shoud have the same shape as self.y_pred',
                    ('y', y.type, 'y_pred', self.y_pred.type)
            )
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


def load_data():
    mnist = input_data.read_data_sets("/tmp/data/")

    print 'Reading data set complete'

    def shared_dataset(data_xy, borrow=True):
        '''
        Function that loads the dataset into shared variables
        '''
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset((mnist.test.images,
                                             mnist.test.labels))
    train_set_x, train_set_y = shared_dataset((mnist.train.images,
                                               mnist.train.labels))
    valid_set_x, valid_set_y = shared_dataset((mnist.validation.images,
                                               mnist.validation.labels))
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000, batch_size=600):
    '''
    Demonstrate stochastic gradient descent optimization of a log-linear model

    @learning_rate
    -type : float
    -param : learning rate used

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

    classifier = NeuralNetwork(input=x, n_in=28*28, n_out=10)

    cost = classifier.cross_entropy(y)

    test_model = theano.function(
            inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]
            }
    )

    validate_model = theano.function(
            inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]
            }
    )

    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    train_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
    )
    
    print '... training the model'

    patience = 5000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = np.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = [validate_model(i)
                                     for i in xrange(n_valid_batches)]
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
                    if this_validation_loss < best_validation_loss * \
                            improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                        
                    best_validation_loss = this_validation_loss

                    test_losses = [test_model(i)
                                   for i in xrange(n_test_batches)]
                    test_score = np.mean(test_losses)

                    print(
                        (
                            '   epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

                    with open('best_model_nn.pkl', 'w') as f:
                        cPickle.dump(classifier, f)

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print 'The code run for %d epochs, with %f epochs/sec' % (
            epoch, 1. * epoch / (end_time - start_time))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.1fs' % ((end_time - start_time)))
    
def predict():
    classifier = cPickle.load(open('best_model_nn.pkl'))

    predict_model = theano.function(
            inputs=[classifier.input],
            outputs=classifier.y_pred)

    datasets = load_data()
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()

    predicted_values = predict_model(test_set_x[:10])
    print ("Predicted values for the first 10 examples in test set:")
    print predicted_values

if __name__ == '__main__':
    sgd_optimization_mnist()
