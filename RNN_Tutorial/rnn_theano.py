import numpy as np
import theano
import theano.tensor as T
from util import *
import operator

class RNN:

    def __init__(self,word_dim,hidden_dim=100,bptt_truncate=4):
        # Assign instance varibles
        self.word_dim=word_dim
        self.hidden_dim=hidden_dim
        self.bptt_truncate=bptt_truncate
        # Randomly initialize the network parameters
        U=np.random.uniform(-np.sqrt(1./word_dim),np.sqrt(1./word_dim),(hidden_dim,word_dim))
        V=np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(word_dim,hidden_dim))
        W=np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(hidden_dim,hidden_dim))
        # Theano: Created shared varibles
        self.U=theano.shared(name='U',value=U.astype(theano.config.floatX))
        self.V=theano.shared(name='V',value=V.astype(theano.config.floatX))
        self.W=theano.shared(name='W',value=W.astype(theano.config.floatX))
        self.params=[self.U,self.V,self.W]
        self.build()

    def build(self):
        x=T.ivector('x')
        y=T.ivector('y')
        lr=T.scalar('learning_rate')

        def _recurrence(x_t,s_tm1):
            s_t=T.tanh(self.U[:,x_t]+T.dot(s_tm1,self.W))
            o_t=T.nnet.softmax(T.dot(s_t,self.V))
            return [o_t[0],s_t]

        [o,s],updates=theano.scan(fn=_recurrence,
                                  sequences=x,
                                  outputs_info=[None,dict(initial=T.zeros(self.hidden_dim))],
                                  truncate_gradient=self.bptt_truncate,
                                  strict=True)
        prediction=T.argmax(o,axis=1)
        o_error=T.sum(T.nnet.categorical_crossentropy(o,y))

        # Gradients
        gparams=T.grad(o_error,self.params)
        updates=[(param,param-lr*gparam) for param,gparam in zip(self.params,gparams)]


        # Assign functions
        self.forward_propagation=theano.function([x],o)
        self.predict=theano.function([x],prediction)
        self.train=theano.function(intputs=[x,y,lr],
                                   outputs=o_error,
                                   updates=updates)
    def calculate_total_loss(self,X,Y):
        return np.sum([self.ce_error(x,y) for x,y in zip(X,Y)])

    def calculate_loss(self,X,Y):
        # Divide calculate_loss by the number of words
        num_words=np.sum([len(y) for y in Y])
        return self.calculate_total_loss(X,Y)/float(num_words)

def gradient_check_theano(model, x, y, h=0.001, error_threshold=0.01):
    # Overwrite the bptt attribute. We need to backpropagate all the way to get the correct gradient
    model.bptt_truncate = 1000
    # Calculate the gradients using backprop
    bptt_gradients = model.bptt(x, y)
    # List of all parameters we want to chec.
    model_parameters = ['U', 'V', 'W']
    # Gradient check for each parameter
    for pidx, pname in enumerate(model_parameters):
        # Get the actual parameter value from the mode, e.g. model.W
        parameter_T = operator.attrgetter(pname)(model)
        parameter = parameter_T.get_value()
        print "Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape))
        # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
        it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            # Save the original value so we can reset it later
            original_value = parameter[ix]
            # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
            parameter[ix] = original_value + h
            parameter_T.set_value(parameter)
            gradplus = model.calculate_total_loss([x],[y])
            parameter[ix] = original_value - h
            parameter_T.set_value(parameter)
            gradminus = model.calculate_total_loss([x],[y])
            estimated_gradient = (gradplus - gradminus)/(2*h)
            parameter[ix] = original_value
            parameter_T.set_value(parameter)
            # The gradient for this parameter calculated using backpropagation
            backprop_gradient = bptt_gradients[pidx][ix]
            # calculate The relative error: (|x - y|/(|x| + |y|))
            relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
            # If the error is to large fail the gradient check
            if relative_error > error_threshold:
                print "Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix)
                print "+h Loss: %f" % gradplus
                print "-h Loss: %f" % gradminus
                print "Estimated_gradient: %f" % estimated_gradient
                print "Backpropagation gradient: %f" % backprop_gradient
                print "Relative Error: %f" % relative_error
                return
            it.iternext()
        print "Gradient check for parameter %s passed." % (pname)
