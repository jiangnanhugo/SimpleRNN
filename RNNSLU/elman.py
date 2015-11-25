# coding=utf-8
import theano
import numpy
import os

from theano import tensor as T
from collections import OrderedDict

class model(object):
    
    def __init__(self, nh, nc, ne, de, cs):
        '''
        nh :: dimension of the hidden layer
        nc :: number of classes. RNN 的输出是其标对应label的概率，有多少种label classes，输出层就有多少结点.
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        cs :: word window context size 
        '''
        # parameters of the model
        #embedding 矩阵一行代表一个word的feature representation，行的长度就是feature的维度
        self.emb = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (ne+1, de)).astype(theano.config.floatX)) # add one for PADDING at the end
        #输入是context feature，所以一行的维度为 word embeddings dimension * context size
        self.Wx  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (de * cs, nh)).astype(theano.config.floatX))
        self.Wh  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nh, nh)).astype(theano.config.floatX))
        self.W   = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nh, nc)).astype(theano.config.floatX))
        self.bh  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        self.b   = theano.shared(numpy.zeros(nc, dtype=theano.config.floatX))
        self.h0  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))

        # bundle
        self.params = [ self.emb, self.Wx, self.Wh, self.W, self.bh, self.b, self.h0 ]
        self.names  = ['embeddings', 'Wx', 'Wh', 'W', 'bh', 'b', 'h0']
        # idx 在这里代替input words,作为一个占位符
        idxs = T.imatrix() # as many columns as context window size/lines as words in the sentence
        #取idx对应的单词的词向量，再拼接成 context representation,再reshape其矩阵维度 ，
        # 一行的维度 dimension of word embedding * context size,
        #行数为 idx的单词数量
        x = self.emb[idxs].reshape((idxs.shape[0], de*cs))
        y    = T.iscalar('y') # label

        '''
        input layer: x, dimension: de*cs
        hidden layer: h, dimension: nh
        output layer: s, dimension: nc
        h(t)=sigmoid(x(t)*Wx+h(t-1)*Wh+bh);
        s(t)=softmax(h(t)*W+b)
        '''
        def recurrence(x_t, h_tm1):
            h_t = T.nnet.sigmoid(T.dot(x_t, self.Wx) + T.dot(h_tm1, self.Wh) + self.bh)
            s_t = T.nnet.softmax(T.dot(h_t, self.W) + self.b)
            return [h_t, s_t]

        # 调用fn函数，遍历sequence，函数输出为output_info.
        [h, s], _ = theano.scan(fn=recurrence, \
            sequences=x, outputs_info=[self.h0, None], \
            n_steps=x.shape[0])

        p_y_given_x_lastword = s[-1,0,:]
        p_y_given_x_sentence = s[:,0,:]
        #对于不同的label classes，挑选最有可能的label，选概率最大的那个下标，即为 prediction = argmax(y)
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)

        # cost and gradients and learning rate
        # lr :learning rate
        lr = T.scalar('lr')
        # negative log likehood
        nll = -T.log(p_y_given_x_lastword)[y]
        # 自动求导
        gradients = T.grad( nll, self.params )
        # params=params-learning_rate*gradients
        updates = OrderedDict(( p, p-lr*g ) for p, g in zip( self.params , gradients))
        
        # theano functions
        self.classify = theano.function(inputs=[idxs], outputs=y_pred)

        self.train = theano.function( inputs  = [idxs, y, lr],
                                      outputs = nll,
                                      updates = updates )

        #每次迭代完emb之后，要对其进行归一化。
        self.normalize = theano.function( inputs = [],
                         updates = {self.emb:\
                         self.emb/T.sqrt((self.emb**2).sum(axis=1)).dimshuffle(0,'x')})

    def save(self, folder):   
        for param, name in zip(self.params, self.names):
            numpy.save(os.path.join(folder, name + '.npy'), param.get_value())
