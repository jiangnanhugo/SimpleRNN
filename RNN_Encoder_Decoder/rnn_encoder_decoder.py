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
        n_dict=None,n_hidden=200,
        We=None,Whx=None,Whx2=None,Whh=None,Why=None,bh=None,b=None):

        self.n_hidden = n_hidden
        self.n_dict = n_dict
        self.input_size=input_size


        if not We:
            initial_We=numpy.asarray(
                numpy_rng.uniform(
                    low=-1*numpy.sqrt(6./(self.n_hidden+self.n_dict)),
                    high=1*numpy.sqrt(6./(self.n_hidden+self.n_dict)),
                    size=(self.n_dict,self.n_hidden)),
                dtype=theano.config.floatX)
            self.We=theano.shared(value=initial_We,name="word_vectors",borrow=True)

        if not Whx:
            initial_Whx=numpy.asarray(
                numpy_rng.uniform(
                    low=-1*numpy.sqrt(6./(self.n_hidden+self.n_hidden)),
                    high=1*numpy.sqrt(6./(self.n_hidden+self.n_hidden)),
                    size=(self.n_hidden,self.n_hidden)),
                dtype=theano.config.floatX)
            self.Whx=theano.shared(value=initial_Whx,name='Wx',borrow=True)

        if not Whx2:
            initial_Whx2=numpy.asarray(
                numpy_rng.uniform(
                    low=-1*numpy.sqrt(6./(self.n_hidden+self.n_hidden)),
                    high=1*numpy.sqrt(6./(self.n_hidden+self.n_hidden)),
                    size=(self.n_hidden,self.n_dict)),
                dtype=theano.config.floatX)
            self.Whx2=theano.shared(value=initial_Whx2,name='Whx2',borrow=True)

        if not Whh:
            initial_Whh = numpy.asarray(numpy_rng.uniform(
                    low=-1 * numpy.sqrt(6. / (self.n_hidden+self.n_hidden)),
                    high=1 * numpy.sqrt(6. / (self.n_hidden+self.n_hidden)),
                    size=(2,self.n_hidden, self.n_hidden)),
                dtype=theano.config.floatX)
            self.Whh = theano.shared(value=initial_Whh, name='Hidden', borrow=True)

        if not Why:
            initial_Why = numpy.asarray(numpy_rng.uniform(
                    low=-1 * numpy.sqrt(6. / (self.n_hidden+self.n_hidden)),
                    high=1 * numpy.sqrt(6. / (self.n_hidden+self.n_hidden)),
                    size=(self.n_dict, self.n_hidden)),
                dtype=theano.config.floatX)
            self.Why = theano.shared(value=initial_Why, name='Hidden', borrow=True)

        if not bh:
            initial_bh = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (self.n_dict)),
                    high=4 * numpy.sqrt(6. / (self.n_dict)),
                    size=(2,self.n_hidden,)),
                dtype=theano.config.floatX)
            self.bh = theano.shared(value=initial_bh, name='b', borrow=True)

        if not b:
            initial_b = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (self.n_hidden)),
                    high=4 * numpy.sqrt(6. / (self.n_hidden)),
                    size=(self.n_dict,)),
                dtype=theano.config.floatX)
            self.b = theano.shared(value=initial_b, name='bh', borrow=True)

        self.params = [self.We,self.Whx,self.Whh,self.Why, self.b,self.bh]


    def get_params(self):
        return {'We':self.We,'Whx':self.Whx,'Whh': self.Whh, 'Why': self.Why, 'bh': self.bh, 'b': self.b}

    def encode(self, sentence):
        h0 = theano.shared(value=np.zeros((self.n_hidden), dtype=theano.config.floatX))

        def _step(x_t, h_tm1):
            x_e=self.We[x_t,:]
            h_t = sigmoid(T.dot(self.Whh[0], h_tm1) + T.dot(self.Whx,x_e) +self.bh[0])
            return h_t

        h, _ = theano.scan(fn = _step,
                           sequences = sentence,
                           outputs_info = h0)

        return h[-1]

    def decode(self, vector_rep):

        h0=vector_rep
        a0=T.dot(self.Why , h0) + self.b
        y0=T.reshape(softmax(a0),a0.shape)

        def _step(h_tm1,y_tm1):
            h_t = sigmoid(T.dot(self.Whx2 , y_tm1) + T.dot(self.Whh[1], h_tm1) + self.bh[1])
            a=T.dot(self.Why, h_t)+self.b
            y_t=T.reshape(softmax(a),a.shape)
            return [h_t,y_t]

        [outputs, hidden_state], _ = theano.scan(fn = _step,
                                                 outputs_info = [h0,y0],
                                                 n_steps = self.input_size)
        y_pred= T.argmax(outputs,axis=1)
        return outputs,y_pred

    def get_encode(self,lr,x):
        context=self.encode(x)
        return context

    def get_decode(self,lr,x):
        context=self.encode(x)
        output,y_pred=self.decode(context)
        return output,y_pred


    def get_cost_updates(self, lr,x):
        context = self.encode(x)
        output,y_pred= self.decode(context)
        #output_hidden = self.encode(output)
        #alpha=0.1
         #T.mean(T.square(output_hidden - context))+ alpha*
        cost =T.sum(T.nnet.categorical_crossentropy(output, x))

        # calculate the gradient
        gparams = T.grad(cost, self.params)
        updates=[(param, param - lr * gparam) for param, gparam in zip(self.params, gparams)]
        return (cost, updates, output)



def build(src_filename, delimiter=',', header=True, quoting=csv.QUOTE_MINIMAL):
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
        rae = RAE(numpy_rng=rng,
                  n_hidden=50,
                  n_dict=vocabulary_size+1,
                  input_size=input_size)

    return rae

def train_dA(lr=0.1, training_epochs=15, params_dict = False, print_every = 100,
            data=None):

    x = T.lvector('x')
    input_size = T.scalar(dtype='int64')
    dA = make_dA(params=params_dict, input_size=input_size, data=x)
    #cost, updates, output = dA.get_cost_updates(lr=lr,x=x)
    output,y_pred=dA.get_decode(lr,x)

    model=theano.function(
        [x],
        [output,y_pred],
        givens={input_size:x.shape[0]}
    )
    '''
    model = theano.function(
        [x],
        [cost,output],
        updates=updates,
        givens={input_size: x.shape[0]}
    )
    '''

    start_time = time.clock()
    for epoch in xrange(training_epochs):
        cost_history = []
        for index in range(len(data)):
            #cost,predict= model(np.asarray(data[index]))
            output,y_pred=model(np.asarray(data[index]))
            print output
            print y_pred
            #cost_history.append(cost)
            #if index % print_every == 0:
            #    print 'Iteration %d, cost %f' % (index, cost)
                #print predict
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
    X_train = np.asarray(pickle.load(f))
    theano.compile.mode.Mode(linker='cvm', optimizer='fast_run')

    cost_history, params, model = train_dA(data=X_train,
                                           training_epochs=10,
                                           lr=0.1,
                                           params_dict=False)
    parameter_file = open("parameters.pkl", 'w')
    pickle.dump(params, parameter_file)

    #test_sentences=map(lambda x: np.array(x), sentences[20000:21000])
    #test_dA(model,data=test_sentences)
