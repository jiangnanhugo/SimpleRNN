import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import matplotlib
import theano
import theano.tensor as T
from IPython.display import Image
from IPython.display import SVG
import timeit

# Display plots inline and change default figure size
#matplotlib inline
matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)


# Generate a dataset and plot it
np.random.seed(0)
train_X,train_y=sklearn.datasets.make_moons(200,noise=0.2)
train_X=train_X.astype(np.float32)
train_y=train_y.astype(np.int32)
plt.scatter(train_X[:,0],train_X[:,1],s=40,c=train_y,cmap=plt.cm.Spectral)
X=T.matrix("X") # matrix of doubles
y=T.lvector('y') # vector of int64

num_examples=len(train_X)
nn_input_dim=2 # input layer dimensionality
nn_output_dim=2 # output layer dimensionality
nn_hdim=100 # hidden layer dimensionality

# Gradient descent parameters (I picked these by hand)
epsilon=0.01 # learning rate for gradient de3scent
reg_lambda=0.01 # regularization strength




W1=theano.shared(np.random.randn(nn_input_dim,nn_hdim),name='W1')
b1=theano.shared(np.zeros(nn_hdim),name='b1')
W2=theano.shared(np.random.randn(nn_hdim,nn_output_dim),name='W2')
b2=theano.shared(np.zeros(np.zeros(nn_output_dim),name='b2'))

# Forward propagation
# Note: We are just defining the expressions, nothing is evaluated here!
z1=X.dot(W1)+b1
a1=T.tanh(z1)
z2=a1.dot(W2)+b2
y_hat=T.nnet.softmax(z2) #output probabities

# The regularization term
loss_reg=1./num_examples*reg_lambda/2*(T.sum(T.sqr(W1)))+T.sum(T.sqr(W2))
# the loss function we want to optimize
loss=T.nnet.categorical_crossentropy(y_hat,y).mean()+loss_reg

# Returns a class prediction
prediction=T.argmax(y_hat,axis=1)


forward_prop=theano.function([X],y_hat)
calculate_loss=theano.function([X,y],loss)
predict=theano.function([X],prediction)

forward_prop([[1,2]])
# Backpropagation (Manual)
# Note: we are just defining the expressions, nothing is evaluated here!
# y_onehot=T.eye(2)[y]
# delta3=y_hat-y_onehot
# dW2=(a1.T).dot(delta3)*(1.+reg_lambda)
# db2=T.sum(delta3,axis=0)
# delta2=delta3.dot(W2.T)*(1.-T.sqr(a1))
# dW1=T.dot(X.T,delta2)*(1+reg_lambda)
# db1=T.sum(delta2,axis=0)
# Easy: Let Theano calculate the derivatives for us!
dW2 = T.grad(loss, W2)
db2 = T.grad(loss, b2)
dW1 = T.grad(loss, W1)
db1 = T.grad(loss, b1)

gradient_step=theano.function([X,y],updates=((W2,W2-epsilon*dW2),
                                             (W1,W1-epsilon*dW1),
                                             (b2,b2-epsilon*db2),
                                             (b1,b1-epsilon*db1)))

def build_model(num_passes=20000,print_loss=False):
    # Re-Initialize the parameters to random values. We need to learn these.
    # (Need in case we call this function multiole times)
    np.random.seed(0)
    W1.set_value(np.random.randn(nn_input_dim,nn_hdim)/np.sqrt(nn_input_dim))
    b1.set_value(np.zeros(nn_hdim))
    W2.set_value(np.random.randn(nn_hdim,nn_output_dim)/np.sqrt(nn_hdim))
    b2.set_value(np.zeros(nn_output_dim))

    for i in xrange(0,num_passes):
        gradient_step(train_X,train_y)


build_model(print_loss=True)

