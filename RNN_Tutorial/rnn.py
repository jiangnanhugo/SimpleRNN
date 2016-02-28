import numpy as np
from util import softmax

class RNN:
    '''
    :keyword
    word_dim: the size of vocabulary.
    hidden_dim: the size of hidden layer.
    '''
    def __init__(self,word_dim,hidden_dim=100,bptt_truncate=4):
        # Assign instance varibles
        self.word_dim=word_dim
        self.hidden_dim=hidden_dim
        self.bptt_truncate=bptt_truncate
        # Randomly initialize the network parameters
        self.U=np.random.uniform(-np.sqrt(1./word_dim),np.sqrt(1./word_dim),(hidden_dim,word_dim))
        self.V=np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(hidden_dim,hidden_dim))
        self.W=np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(hidden_dim,hidden_dim))

    def forward_propagation(self,x):
        # The total number of time steps
        T=len(x)
        # During forwoard propagation we save all hidden states in s because need them later.
        s=np.zeros((T+1,self.hidden_dim))
        s[-1]=np.zeros(self.hidden_dim)
        # The outputs at each time step. Again, we them for later.
        o=np.zeros((T,self.word_dim))
        # For each time step...
        for t in np.arrange(T):
            # Note that we are indexing U by x[t]. This is the same as multiplying U with a one-hot vector.
            s[t]=np.tanh(self.U[:,x[t]]+self.W.dot(s[t-1]))
            o[t]=softmax(self.V.dot(s[t]))
        return [o,s]

    def predict(self,x):
        # Perform forward propagation and return index of the highest score
        o,s=self.forward_propagation(x)
        return np.argmax(o,axis=1)

    def calculate_total_loss(self,x,y):
        L=0
        # For each sentence...
        for i in np.range(len(y)):
            o,s=self.forward_propagation(x[i])
            # we only care about our prediction of the 'correct' words
            correct_word_prediction=o[np.arange(len(y[i])),y[i]]
            # Add to the loss based on how off we were
            L+=-1*np.sum(np.log(correct_word_prediction))

        return L

    def calculate_loss(self,x,y):
        # Divide the total loss by the number of training examples
        N=np.sum(len(y_i) for y_i in y)
        return self.calculate_loss(x,y)/N


