import nltk
import sys
import os
import csv
import time
import itertools
from datetime import datetime
from util import *
from rnn_theano import RNN

vocabulary_size=8000
hidden_dim=80
learning_rate=0.005
nepoch=100
sentence_start_token='SENTENCE_START'
sentence_end_token  ='SENTENCE_END'
unknown_token="UNKNOWN_TOKEN"
model_file=''

def train_with_sgd(model,X_train,y_train,learning_rate=0.005,nepoch=1,evaluate_loss_after=5):
    # We keep track of lossed so we can plot them later
    losses=[]
    num_examples_seen=0
    for epoch in range(nepoch):
        # Optionally evaluate the loss
        if(epoch % evaluate_loss_after==0):
            loss=model.calculate_loss(X_train,y_train)
            losses.append((num_examples_seen,loss))
            time=datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            print "%s: Loss after num_examples_seen=%d epoch=%d: %f" %(time,num_examples_seen,epoch,loss)
            # Adjust the learning rate if loss increases
            if (len(losses)>1 and losses[-1][1]>losses[-2][1]):
                learning_rate=learning_rate*0.5
                print "Setting learning rate to %f" % learning_rate
            sys.stdout.flush()
            # Saving model parameters
            save_model_parameters_theano("./data/rnn-theano-%d-%d-%s.npz" %(model.hidden_dim,model.word_dim,time),model)
        # For each training examples...
        for i in range(len(y_train)):
            # One SGD Step
            model.sgd_step(X_train[i],y_train[i],learning_rate)
            num_examples_seen+=1

print "Reading CSV file..."
with open('reddit.csv') as f:
    reader=csv.reader(f,skipinitialspace=True)
    reader.next()
    # Split full comments into sentences
    sentences=itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
    # Append SENTENCE_START and SENTENCE_END
    sentences=["%s %s %s" % (sentence_start_token,x,sentence_end_token) for x in sentences]
print "Parsed %d sentences." %(len(sentences))

# Tokenize the sentences into words
tokenized_sentences=[nltk.word_tokenize(sent) for sent in sentences]

# Count the word frequencies
word_freq=nltk.FreqDist(itertools.chain(*tokenized_sentences))
print "Founded %d unique words tokens." % len(word_freq.items())

# Get the most common words and build index2word and word2index vectors
vocab=word_freq.most_common(vocabulary_size-1)
index2word=[x[0] for x in vocab]
index2word.append(unknown_token)
word2index=dict([(w,i) for i,w in enumerate(index2word)])

print "using vocabulary size %d." % vocabulary_size
print "The least frequent word in out vocabulay is '%s' and apeared %d times." %(vocab[-1][0],vocab[-1][1])

# Replace all owrds not in our vocabulary with the unkwnon token
for i,sent in enumerate(tokenized_sentences):
    tokenized_sentences[i]=[w if w in word2index else unknown_token for w in sent]

# Create the training data
X_train=np.asarray([[word2index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train=np.asarray([[word2index[w] for w in sent[1:]] for sent in tokenized_sentences])




model=RNN(vocabulary_size,hidden_dim=hidden_dim)
t1=time.time()
model.sgd_step(X_train[10],y_train[10],learning_rate)
t2=time.time()
print "SGD Step time: %f milliseconds" %((t2-t1)*1000.)
train_with_sgd(model,X_train,y_train,nepoch=nepoch,learning_rate=learning_rate)