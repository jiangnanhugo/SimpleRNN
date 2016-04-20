import csv
import pickle
import time
import random

import numpy
import numpy as np

import theano
import theano.tensor as T
from theano.tensor.nnet import sigmoid,softmax


vocabulary_size=2000
class RAE(object):
    def __init__(self,numpy_rng,input_size=None,
        n_dict=None,input=None,n_hidden=200,
        Wv=None,Wx=None,H=None,C=None,Y=None,S=None,b=None
    ):

        self.n_hidden = n_hidden
        self.n_dict = n_dict
        self.x = input

        if not Wv:
            initial_Wv=numpy.asarray(
                numpy_rng.uniform(
                    low=-1*numpy.sqrt(6./(self.n_hidden+self.n_dict)),
                    high=1*numpy.sqrt(6./(self.n_hidden+self.n_dict)),
                    size=(self.n_dict,self.n_hidden)),
                dtype=theano.config.floatX)
            Wv=theano.shared(value=initial_Wv,name="word_vectors",borrow=True)

        if not Wx:
            initial_Wx=numpy.asarray(
                numpy_rng.uniform(
                    low=-1*numpy.sqrt(6./(self.n_hidden+self.n_hidden)),
                    high=1*numpy.sqrt(6./(self.n_hidden+self.n_hidden)),
                    size=(2,self.n_hidden,self.n_hidden)),
                dtype=theano.config.floatX)
            Wx=theano.shared(value=initial_Wx,name='Wx',borrow=True)

        if not H:
            initial_H1 = numpy.asarray(numpy_rng.uniform(
                    low=-1 * numpy.sqrt(6. / (self.n_hidden+self.n_hidden)),
                    high=1 * numpy.sqrt(6. / (self.n_hidden+self.n_hidden)),
                    size=(2,self.n_hidden, self.n_hidden)),
                dtype=theano.config.floatX)
            H = theano.shared(value=initial_H1, name='Hidden', borrow=True)

        if not b:
            initial_b = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (self.n_dict)),
                    high=4 * numpy.sqrt(6. / (self.n_dict)),
                    size=(self.n_dict,)),
                dtype=theano.config.floatX)
            b = theano.shared(value=initial_b, name='b', borrow=True)

        self.Wv=Wv
        self.Wx=Wx
        self.H = H
        #self.C = C
        self.Y = Y
        self.S = S
        self.b=b



        self.params = [self.Wv,self.Wx,self.H, self.b]


    def get_params(self):
        return {'Wv':self.Wv,'Wx':self.Wx,'H': self.H, 'C': self.C, 'Y': self.Y, 'S': self.S}

    def encode(self, sentence):
        h0 = theano.shared(value=np.zeros((self.n_hidden), dtype=theano.config.floatX))

        def _encode_recurrence(x_t, h_tm1):
            x_e=self.Wv[x_t,:]
            h_t = sigmoid(T.dot(self.H[0], h_tm1) + T.dot(x_e,self.Wx))
            return h_t

        h, _ = theano.scan(fn = _encode_recurrence,
                           sequences = sentence,
                           outputs_info = h0)

        return h[-1]

    def decode(self, vector_rep):

        def _decode_recurrence(x_t,h_tm1):
            x_e=self.Wv[x_t,:]
            h_t = sigmoid(T.dot(self.H[1], h_tm1) + T.dot(x_e,self.Wx[1]))
            s_t = softmax(T.dot(self.S, h_t)+self.b)
            return [h_t,s_t]

        [h, s], _ = theano.scan(fn = _decode_recurrence,
                                sequences=self.x,
                                outputs_info = [vector_rep],
                                n_steps = self.x.shape[0])
        y= T.argmax(s,axis=1)
        return y,s


    def get_cost_updates(self, lr):
        context = self.encode(self.x)
        output,softmaxes= self.decode(context)
        output_hidden = self.encode(output)
        alpha=0.1
        cost = T.mean(T.square(output_hidden - context))+ alpha*T.sum(T.nnet.categorical_crossentropy(softmaxes, self.x))

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
    #glove_matrix, glove_vocab, _ = build('glove.6B.50d.txt', delimiter=' ', header=False, quoting=csv.QUOTE_NONE)
    #glove_matrix = theano.shared(value=np.array(glove_matrix, dtype=theano.config.floatX), borrow=True)
    rng = numpy.random.RandomState(123)

    if params == False:
        rae = RAE(numpy_rng=rng,input=data,
            n_hidden=50,
            n_dict=vocabulary_size+1,
            input_size=input_size)
    else:
        rae = RAE(numpy_rng=rng,
            input=data,
            n_hidden=50,
            n_dict=vocabulary_size+1,
            input_size=input_size,
            Wv=params["Wv"],
            Wx=params['Wx'],
            H=params['H'],
            C=params['C'],
            Y=params['Y'],
            S=params['S'])
    return rae

def train_dA(lr=0.1, training_epochs=15, params_dict = False, print_every = 100,
            data=None):

    x = T.lvector('x')
    input_size = T.scalar(dtype='int64')
    dA = make_dA(params=params_dict, input_size=input_size, data=x)
    cost, updates, output = dA.get_cost_updates(lr=lr)

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
            cost, predict= model(data[index])
            cost_history.append(cost)
            if index % print_every == 0:
                print 'Iteration %d, cost %f' % (index, cost)
                print predict
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
    f = open("X_train.pkl", 'r')
    X_train = pickle.load(f)
    theano.compile.mode.Mode(linker='cvm', optimizer='fast_run')

    cost_history, params, model = train_dA(data=X_train,
                                           training_epochs=10,
                                           lr=0.1,
                                           params_dict=False)
    parameter_file = open("parameters.pkl", 'w')
    pickle.dump(params, parameter_file)

    #test_sentences=map(lambda x: np.array(x), sentences[20000:21000])
    #test_dA(model,data=test_sentences)
