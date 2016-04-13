import numpy
import theano
import theano.tensor as T
rng=numpy.random

N=400
feats=784
D=(rng.randn(N,feats),rng.randint(size=N,low=0,high=2))
training_steps=10000

# Declare Theano symbolic varibles
x=T.matrix("x")
y=T.vector('y')
w=theano.shared(value=rng.randn(feats),
                name="W",
                borrow=True)
b=theano.shared(value=0.,
                name='b')
params=[w,b]
print("initial model:",w.shape,b.shape)

# Construct Theano expression graph
p=1./(1.+T.exp(--T.dot(x,w)-b))
prediction=p>0.5
xent=T.mean(-y*T.log(p)-(1-y)*T.log(1-p))
cost=xent+0.01*(w**2).sum()
gparams=T.grad(cost,params)
updates=[(pa,pa-0.1*gpa) for pa, gpa in zip(params,gparams)]
model=theano.function(
    inputs=[x,y],
    outputs=[prediction,cost],
    updates=updates
)
predict=theano.function(
    inputs=[x],
    outputs=prediction
)

nb_epoches=1
batch_size=100
train_error=[]
for epoch in xrange(nb_epoches):
    for i in xrange(training_steps):
        pred,err=model(D[0],D[1])
        train_error.append(err)
        if i %batch_size==0:
            print numpy.mean(train_error)
            train_error=[]

print("Final model:")
print(w.get_value())
print(b.get_value())
print("target values for D:")
print(D[1])
print("prediction on D:")
print(predict(D[0]))

