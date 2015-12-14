import copy ,numpy as np
np.random.seed(0)

# compute sigmoid nonlinearity
# sigmoid(x)=1/(1+e^{-x})
def sigmoid(x):
    output=1/(1+np.exp(-x))
    return output


# convert output of sigmoid function to this dervative
def sigmoid_grad(output):
    return output*(1-output)

# training dataset generation
'''
This is a lookup table: maps integers to its binary code.
converting integers to bit strings.
dict key:number,value:binary value
'''
int2binary={}
# maximum length of the binary numbers we will be adding.
binary_dim=8

largest_number=pow(2,binary_dim)
# np.unpackbits : uint8 ==> 01 string

binary=np.unpackbits(np.array([range(largest_number)],dtype=np.uint8).T,axis=1)

for i in range(largest_number):
    int2binary[i]=binary[i]


alpha=0.1      # learning rate
# we are adding two numbers together, so feed in 2 bits string at each time.
input_dim=2    # input dimension
hidden_dim=16  # hidden dimension
# we only predict the sum, thus we only need one output.
output_dim=1   # output dimension

# initalize neural network weights
synapse_0=2*np.random.random((input_dim,hidden_dim))-1   # input--> hidden
synapse_1=2*np.random.random((hidden_dim,output_dim))-1  # hidden--> output
synapse_h=2*np.random.random((hidden_dim,hidden_dim))-1  # prev_hidden--> hidden, hidden --> next_hidden

'''
init weight update
after we've accumulated several weight updates, we'll update the matrices.
'''
synapse_0_update=np.zeros_like(synapse_0)
synapse_1_update=np.zeros_like(synapse_1)
synapse_h_update=np.zeros_like(synapse_h)


# training logic
for i in range(100000):

    # we are going to generate a simple addition problem (a+b=c)
    # init a number between [0,largest_number/2], larger number will overflow.
    a_int=np.random.randint(largest_number/2) # int version
    a=int2binary[a_int]                       # corresponding binary form

    b_int=np.random.randint(largest_number)/2 # int version
    b=int2binary[b_int]                       # binary encoding


    c_int=a_int+b_int                         # correct answers.
    c=int2binary[c_int]                       # convert the true answer to its binary representation.

    # where we'll store our best guess (binary encoded)
    # the RNN's predictions.
    d=np.zeros_like(c)

    # which we use it as a measure to track convergence.
    overallError=0

    layer_2_deltas=list()                    # layer 2 derivatives
    layers_1_values=list()                   # layer 1 values
    # time 0 has no previous hidden layer, so we initialize one that's off.
    layers_1_values.append(np.zeros(hidden_dim))

    # this loop add bits from position_0 to position_7
    for position in range(binary_dim):
        '''
        generate input and output
        X is layer_0;
        X is a list of 2 bits, one from 'a', one from 'b'.
        it's indexed according to 'position', but it goes right -> left.
        '''
        X=np.array([[ a[binary_dim-position-1],b[binary_dim-position-1] ]])
        # the value of the correct answer.
        y=np.array([[c[binary_dim-position-1]]]).T

        '''
        layer_0: input layer
        layer_1: hidden layer
        layer_2: output layer
        '''
        # hidden layer (input ~+perv_hidden)
        # layers_1_values[-1]: pre_layer_1
        layer_1=sigmoid(
            np.dot(X,synapse_0)+np.dot(layers_1_values[-1],synapse_h)
        )

        # output layer (new binary representation)
        # propagate the hidden layer to output layer to make prediction.
        layer_2=sigmoid(
            np.dot(layer_1,synapse_1)
        )

        # compute the error between prediction and true answers.
        layer_2_error=y-layer_2

        # store the derivative in list, holding the derviate at each time-step.
        layer_2_deltas.append(
            (layer_2_error)*sigmoid_grad(layer_2)
        )

        # calculate the sum of the absolute errors, so that we have a scalar error to track propagation.
        overallError+=np.abs(layer_2_error[0])

        # decode estimate so we print it out
        d[binary_dim-position-1]=np.round(layer_2[0][0])

        # store hidden layer so we can use it in the next timestep
        layers_1_values.append(copy.deepcopy(layer_1))

    '''
    we've done all the forward propagating for all the time steps.
    we will back propagate.
    '''
    future_layer_1_delta=np.zeros(hidden_dim)

    for position in range(binary_dim):
        # index the input data
        X=np.array([[a[position],b[position]]])
        # select the current hidden layer from the list.
        layer_1=layers_1_values[-position-1]
        # select the previous hidden layers.
        pre_layer_1=layers_1_values[-position-2]

        # select the current output error
        layer_2_delta=layer_2_deltas[-position-1]
        '''
        current hidden layer error = next hidden layer error + current output layer error
        '''
        layer_1_delta=\
            (
                future_layer_1_delta.dot(synapse_h.T)
                +layer_2_delta.dot(synapse_1.T)
            )* sigmoid_output_to_dervative(layer_1)

        '''
         construct the weight update, but not actually the weights.
         we will update the weight matrices when we fully bp everything.
        '''
        synapse_1_update+=np.atleast_2d(layer_1).T.dot(layer_2_delta)
        synapse_h_update+=np.atleast_2d(pre_layer_1).T.dot(layer_1_delta)
        synapse_0_update+=X.T.dot(layer_1_delta)

        future_layer_1_delta=layer_1_delta

    # now we have backpropped everything and created our weight updates.
    # it's time to update our weights & empty the update varibles.
    synapse_0+=synapse_0_update*alpha
    synapse_1+=synapse_1_update*alpha
    synapse_h+=synapse_h_update*alpha

    synapse_0_update*=0
    synapse_1_update*=0
    synapse_h_update*=0

    #print out progress
    if(i%1000==0):
        print "Error:"+str(overallError)
        print "Pred:"+str(d)
        print "True:"+str(c)
        out=0
        for index,x in enumerate(reversed(d)):
            out+=x*pow(2,index)
        print str(a_int)+" + "+str(b_int)+" = "+str(out)
        print "------------------"




