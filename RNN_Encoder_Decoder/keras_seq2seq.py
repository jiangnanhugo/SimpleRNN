#-*- coding:utf-8 -*-
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.core import RepeatVector, TimeDistributedDense, Activation
from seq2seq.layers.decoders import LSTMDecoder, LSTMDecoder2, AttentionDecoder
import time
import numpy as np
import cPickle as pickle


maxlen=100
hidden_dim=200
vocab_size=2000

def vectorize_stories(input_list, tar_list, word_idx, input_maxlen, tar_maxlen, vocab_size):

    Y = np.zeros((len(tar_list), tar_maxlen, vocab_size), dtype=np.bool)

    for s_index, tar_tmp in enumerate(tar_list):
        for t_index, token in enumerate(tar_tmp):
            Y[s_index, t_index, token] = 1

    return pad_sequences(input_list, maxlen=input_maxlen), Y

def main():
    f = open("X_train.pkl", 'r')
    X_train = pickle.load(f)
    f=open('word2index.pkl','r')
    word2index=pickle.load(f)
    f=open('index2word.pkl','r')
    index2word=pickle.load(f)

    inputs_train, tars_train = vectorize_stories(X_train, X_train, word2index, maxlen, maxlen, vocab_size)
    X_train=pad_sequences(X_train, maxlen=maxlen)

    decoder_mode = 1  # 0 最简单模式，1 [1]向后模式，2 [2] Peek模式，3 [3]Attention模式
    if decoder_mode == 3:
        encoder_top_layer = LSTM(hidden_dim, return_sequences=True)
    else:
        encoder_top_layer = LSTM(hidden_dim)

    if decoder_mode == 0:
        decoder_top_layer = LSTM(hidden_dim, return_sequences=True)
        decoder_top_layer.get_weights()
    elif decoder_mode == 1:
        decoder_top_layer = LSTMDecoder(hidden_dim=hidden_dim, output_dim=hidden_dim
                                        , output_length=maxlen, state_input=False, return_sequences=True)
    elif decoder_mode == 2:
        decoder_top_layer = LSTMDecoder2(hidden_dim=hidden_dim, output_dim=hidden_dim
                                         , output_length=maxlen, state_input=False, return_sequences=True)
    elif decoder_mode == 3:
        decoder_top_layer = AttentionDecoder(hidden_dim=hidden_dim, output_dim=hidden_dim
                                             , output_length=maxlen, state_input=False, return_sequences=True)

    en_de_model = Sequential()
    en_de_model.add(Embedding(input_dim=vocab_size,
                              output_dim=hidden_dim,
                              input_length=maxlen))
    en_de_model.add(encoder_top_layer)
    if decoder_mode == 0:
        en_de_model.add(RepeatVector(maxlen))
    en_de_model.add(decoder_top_layer)

    en_de_model.add(TimeDistributedDense(vocab_size))
    en_de_model.add(Activation('softmax'))
    print('Compiling...')
    time_start = time.time()
    en_de_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    time_end = time.time()
    print('Compiled, cost time:%fsecond!' % (time_end - time_start))
    for iter_num in range(5000):
        en_de_model.fit(X_train, X_train, batch_size=3, nb_epoch=1, show_accuracy=True)
        out_predicts = en_de_model.predict(X_train)
        for i_idx, out_predict in enumerate(out_predicts):
            predict_sequence = []
            for predict_vector in out_predict:
                next_index = np.argmax(predict_vector)
                next_token = index2word[next_index]
                predict_sequence.append(next_token)
            print('Target output:', X_train[i_idx])
            print('Predict output:', predict_sequence)

        print('Current iter_num is:%d' % iter_num)

if __name__ == '__main__':
    main()