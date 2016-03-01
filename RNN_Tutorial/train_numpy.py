import csv
import numpy as np
import theano
import theano.tensor as T
import operator
import nltk
import itertools
from datetime import datetime

from rnn_numpy import RNN,train_with_sgd

vocabulary_size=8000
unknown_token='UNKOWN_TOKEN'
sentence_start_token='SENTENCE_START'
sentence_end_token='SENTENCE_END'

#  Read the data and append SENTENCE_START and SENTENCE_END tokens
print "Reading CSV file..."
with open('reddit.csv') as f:
    reader=csv.reader(f,skipinitialspace=True)
    reader.next()
    # Split full comment into sentences
    sentences=itertools.chain(*[nltk.sent_tokenize( x[0].decode('utf-8').lower()) for x in reader])
    # Append SENTENCE_START and SENTENCE_END
    sentences=['%s %s %s'%(sentence_start_token,x,sentence_end_token) for x in sentences]

print 'Parsed %d sentences.' %(len(sentences))

# tokenized the sentences into words
print len(sentences)
tokenized_sentences=[nltk.word_tokenize(sent) for sent in sentences]
print len(tokenized_sentences)
# Count the word frequencies
word_freq=nltk.FreqDist(itertools.chain(*tokenized_sentences))
print "Found %d unique words tokens." % len(word_freq.items())

# Get the most common words and build index_to_word and word_to_index vectors
vocab=word_freq.most_common(vocabulary_size-1)
index2word=[x[0] for x in vocab]
index2word.append(unknown_token)
word2index=dict([(w,i) for i,w in enumerate(index2word)])

print "Using vocabulary size %d." % vocabulary_size
print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0],vocab[-1][1])


# Replace all words not in our vocabulary with the unknown token
for i,sent in enumerate(tokenized_sentences):
    tokenized_sentences[i]=[w if w in word2index else unknown_token for w in sent]


print "\nExample sentence: '%s'" % sentences[0]
print "\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0]
print len(tokenized_sentences)

# Create the training data
X_train=np.asarray([[word2index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train=np.asarray([[word2index[w] for w in sent[1:]] for sent in tokenized_sentences])

# Print an training data example
x_example,y_example=X_train[17],y_train[17]
print "x:\n%s\n%s" %(" ".join([index2word[x] for x in x_example]),x_example)
print "\ny:\n%s\n%s" %(" ".join([index2word[x] for x in y_example]),y_example)

# RNN model
np.random.seed(10)
model=RNN(vocabulary_size)
o,s=model.forward_propagation(X_train[10])
print X_train[10]
print o.shape
print o

predictions=model.predict(X_train[10])
print predictions.shape
print predictions

# Limit to 1000 examples to save time
print "Expected Loss for random predicitons: %f " % np.log(vocabulary_size)
print "Actual loss: %f" % model.calculate_loss(X_train[:1000],y_train[:1000])

np.random.seed(10)
model = RNN(vocabulary_size)
model.sgd_step(X_train[10], y_train[10], 0.005)

np.random.seed(10)
# Train on a small subset of the data to see what happens
model = RNN(vocabulary_size)
losses = train_with_sgd(model, X_train[:100], y_train[:100], nepoch=10, evaluate_loss_after=1)
