import csv
import pickle
import time
import random

import numpy
import numpy as np

import theano
import theano.tensor as T
from theano.tensor.nnet import sigmoid,softmax

class dA(object):
    def __init__(self,numpy_rng,input_size=None,
        n_dict=None,input=None,n_input=50,n_hidden=200,
        Wx=None,H1=None,H2=None,C=None,Y=None,S=None,
        word_vectors=None,
        end_tokens=2
    ):
        self.n_input=n_input
        self.n_hidden = n_hidden
        self.word_vectors = word_vectors
        self.n_dict = n_dict

        self.end_tokens = end_tokens

        if not Wx:
            initial_Wx=numpy.asarray(
                numpy_rng.uniform(
                    low=-1*numpy.sqrt(6./(self.n_hidden+self.n_input)),
                    high=1*numpy.sqrt(6./(self.n_hidden+self.n_input)),
                    size=(self.n_hidden,self.n_input)),
                dtype=theano.config.floatX)
            Wx=theano.shared(value=initial_Wx,name='Wx',borrow=True)
        if not H1:
            initial_H1 = numpy.asarray(numpy_rng.uniform(
                    low=-1 * numpy.sqrt(6. / (self.n_hidden+self.n_hidden)),
                    high=1 * numpy.sqrt(6. / (self.n_hidden+self.n_hidden)),
                    size=(self.n_hidden, self.n_hidden)),
                dtype=theano.config.floatX)
            H1 = theano.shared(value=initial_H1, name='H1', borrow=True)

        if not H2:
            initial_H2 = numpy.asarray(numpy_rng.uniform(
                    low=-1 * numpy.sqrt(6. / (self.n_hidden+self.n_hidden)),
                    high=1 * numpy.sqrt(6. / (self.n_hidden+self.n_hidden)),
                    size=(self.n_hidden, self.n_hidden)),
                dtype=theano.config.floatX)
            H2 = theano.shared(value=initial_H2, name='H2', borrow=True)

        if not C:
            initial_C = numpy.asarray(numpy_rng.uniform(
                    low=-1 * numpy.sqrt(6. / (self.n_hidden+self.n_hidden)),
                    high=1 * numpy.sqrt(6. / (self.n_hidden+self.n_hidden)),
                    size=(self.n_hidden, self.n_hidden)),
                dtype=theano.config.floatX)
            C = theano.shared(value=initial_C, name='C', borrow=True)

        if not Y:
            initial_Y = numpy.asarray(numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (self.n_hidden+self.n_dict)),
                    high=4 * numpy.sqrt(6. / (self.n_hidden+self.n_dict)),
                    size=(self.n_hidden, self.n_dict)),
                dtype=theano.config.floatX)
            Y = theano.shared(value=initial_Y, name='Y', borrow=True)

        if not S:
            initial_S = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (self.n_hidden+self.n_hidden)),
                    high=4 * numpy.sqrt(6. / (self.n_hidden+self.n_hidden)),
                    size=(self.n_dict, self.n_hidden)),
                dtype=theano.config.floatX)
            S = theano.shared(value=initial_S, name='S', borrow=True)

        self.Wx=Wx
        self.H1 = H1
        self.H2 = H2
        self.C = C
        self.Y = Y
        self.S = S

        self.x = input

        self.params = [self.Wx,self.H1, self.S, self.Y, self.C, self.H2]


    def get_params(self):
        return {'Wx':self.Wx,'H1': self.H1, 'H2': self.H2, 'C': self.C, 'Y': self.Y, 'S': self.S}

    def encode(self, sentence):
        h0 = theano.shared(value=np.zeros((self.n_hidden), dtype=theano.config.floatX))

        def _encode_recurrence(x_t, h_tm1):
            h_t = sigmoid(T.dot(self.H1, h_tm1) + T.dot(self.Wx,self.word_vectors[x_t]))
            return h_t

        h, _ = theano.scan(fn = _encode_recurrence,
                           sequences = sentence,
                           outputs_info = h0)

        return h[-1]



    def decode(self, c):
        h0 = sigmoid(T.dot(self.H2, c) + T.dot(self.C, c))
        a0 = T.dot(self.S, h0)
        s0 = T.reshape(T.nnet.softmax(a0), a0.shape)

        def _decode_recurrence(h_tm1, y_tm1, context):
            h_t = sigmoid(T.dot(self.H2, h_tm1) + T.dot(self.Y, y_tm1) + T.dot(self.C, context))
            a = T.dot(self.S, h_t)
            s_t = T.reshape(T.nnet.softmax(a), a.shape)
            return [h_t, s_t]

        [h, s], _ = theano.scan(fn = _decode_recurrence,
                                outputs_info = [h0, s0],
                                non_sequences = c,
                                n_steps = self.x.shape[0])
        y = T.argmax(s, axis=1)

        return y, s





    def get_cost_updates(self, lr, verbose=False, alpha=0):
        context = self.encode(self.x)
        output, softmaxes = self.decode(context)
        output_hidden = self.encode(output)

        if verbose:
            cost_hidden = T.sqrt(T.sum(T.sqr(output_hidden - context)))
            cost_reconstruction = T.sum(T.nnet.categorical_crossentropy(softmaxes, self.x))
            cost_total = cost_hidden + cost_reconstruction
            cost = [cost_hidden, cost_reconstruction, cost_total]
        else:
            cost = T.sqrt(T.sum(T.sqr(output_hidden - context))) + alpha*T.sum(T.nnet.categorical_crossentropy(softmaxes, self.x))

        # calculate the gradient
        gparams = T.grad(cost, self.params)

        updates=[(param, param - lr * gparam) for param, gparam in zip(self.params, gparams)]

        return (cost, updates, output)


def build(src_filename, delimiter=',', header=True, quoting=csv.QUOTE_MINIMAL):
    # Thanks to Prof. Chris Potts, Stanford University, for this function
    reader = csv.reader(file(src_filename), delimiter=delimiter, quoting=quoting)
    colnames = None
    if header:
        colnames = reader.next()
        colnames = colnames[1: ]
    mat = []
    rownames = []
    for line in reader:
        rownames.append(line[0])
        mat.append(np.array(map(float, line[1: ])))
    return (np.array(mat), rownames, colnames)

def make_dA(params=False, data=None, input_size=False):
    glove_matrix, glove_vocab, _ = build('glove.6B.50d.txt', delimiter=' ', header=False, quoting=csv.QUOTE_NONE)
    glove_matrix = theano.shared(value=np.array(glove_matrix, dtype=theano.config.floatX), borrow=True)
    rng = numpy.random.RandomState(123)

    if params == False:
        da = dA(numpy_rng=rng,input=data,
            n_hidden=50,
            word_vectors=glove_matrix,
            n_dict=len(glove_vocab),
            input_size=input_size)
    else:
        da = dA(numpy_rng=rng,
            input=data,
            n_hidden=50,
            word_vectors=glove_matrix,
            n_dict=len(glove_vocab),
            input_size=input_size,
            Wx=params['Wx'],
            H1=params['H1'],
            H2=params['H2'],
            C=params['C'],
            Y=params['Y'],
            S=params['S'])
    return da

def train_dA(lr=0.1, training_epochs=15, params_dict = False, print_every = 100,
            data=None):

    x = T.lvector('x')
    input_size = T.scalar(dtype='int64')
    dA = make_dA(params=params_dict, input_size=input_size, data=x)
    cost, updates, output = dA.get_cost_updates(lr=lr,alpha=0)

    model = theano.function(
        [x],
        [cost, output],
        updates=updates,
        givens={input_size: x.shape[0]}
    )

    start_time = time.clock()
    for epoch in xrange(training_epochs):
        cost_history = []
        for index in range(len(data)):
            cost, _= model(data[index])
            cost_history.append(cost)
            if index % print_every == 0:
                print 'Iteration %d, cost %f' % (index, cost)
        print 'Training epoch %d, cost ' % epoch, numpy.mean(cost_history)

    training_time = (time.clock() - start_time)

    print 'Finished training %d epochs, took %d seconds' % (training_epochs, training_time)

    return cost_history, dA.get_params(), model

def test_dA(model, data=None):
    for index in range(len(data)):
        cost, output = model(data[index])
        print 'Finished testing %d iterations, cost %f' % (index, cost)
        print 'output:',output




if __name__ == '__main__':
    f = open("all_wvi.pkl", 'r')
    sentences = pickle.load(f)
    train_sentences = map(lambda x: np.array(x), sentences[0:2000])
    print train_sentences[2]

    theano.compile.mode.Mode(linker='cvm', optimizer='fast_run')

    cost_history, params, model = train_dA(data=train_sentences,
                                              training_epochs=10,
                                              lr=0.1,
                                              params_dict=False)
    parameter_file = open("parameters.pkl", 'w')
    pickle.dump(params, parameter_file)

    test_sentences=map(lambda x: np.array(x), sentences[20000:21000])
    test_dA(model,data=test_sentences)
