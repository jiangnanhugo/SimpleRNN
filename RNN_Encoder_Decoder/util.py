import csv
import numpy as np
import itertools
import nltk
import cPickle as pickle
import operator

UNKNOWN_TOKEN="UNKNOWN_TOKEN"

def load_data(filename="data/reddit-comments-2015-08.csv",vocabulary_size=2000,min_sent_characters=0):

    # Read the data
    print("Reading CSV file...")
    with open(filename,'rt') as f:
        reader=csv.reader(f,skipinitialspace=True)
        reader.next()
        # Split full comments into sentences
        sentences=itertools.chain(*[nltk.sent_tokenize(x[0].decode("utf-8").lower()) for x in reader])
        # Filter sentences
        sentences=[s for s in sentences if len(s)>=min_sent_characters]
        sentences=[s for s in sentences if "http" not in s]

    print("parsed %d sentences." %(len(sentences)))

    # Tokenize the sentences into words
    tokenized_sentences=[nltk.word_tokenize(sent) for sent in sentences]

    # Count the word frequencies
    word_freq=nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print("Found %d unique word tokens." %len(word_freq.items()))

    # Get the most common words and build index2word and word2index vectors
    vocab=sorted(word_freq.items(),key=lambda x:(x[1],x[0]),reverse=True)[:vocabulary_size-1]
    print ("Using vocabulary size %d." % vocabulary_size)
    print ("The least frequent word in our vocabulary is '%s' and appeared %d times." %(vocab[-1][0],vocab[-1][1]))

    sorted_vocab=sorted(vocab,key=operator.itemgetter(1))
    index2word=[UNKNOWN_TOKEN]+[x[0] for x in sorted_vocab]
    word2index=dict([(w,i) for i,w in enumerate(index2word)])

    # Replace all words not in our vocabulary with the unknown token.
    for i,sent in enumerate(tokenized_sentences):
        tokenized_sentences[i]=[w if w in word2index else UNKNOWN_TOKEN for w in sent]

    # Create the training data
    X_train=[[word2index[w] for w in sent] for sent in tokenized_sentences]

    return X_train,word2index,index2word



if __name__=="__main__":
    X_train,word2index,index2word=load_data()
    with open("X_train.pkl",'w')as f:
        pickle.dump(X_train,f)
    with open("word2index.pkl",'w') as f:
        pickle.dump(word2index,f)
    with open("index2word.pkl",'w')as f:
        pickle.dump(index2word,f)





