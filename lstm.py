from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU

model = Sequential()

model.add(LSTM(input_shape = (emb,),input_dim=emb, output_dim=hidden, return_sequences=True))
model.add(TimeDistributedDense(output_dim=4))
model.add(Activation('softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam')
for seq, label in zip(sequences, y):
   model.train(np.array([seq]), [label])