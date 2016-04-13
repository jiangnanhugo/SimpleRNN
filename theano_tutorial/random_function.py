from theano.tensor.shared_randomstreams import RandomStreams
from theano import function
srng=RandomStreams(seed=234)
rv_u=srng.uniform((2,2))
rv_n=srng.normal((2,2))
f=function(inputs=[],outputs=rv_u)
g=function(inputs=[],outputs=rv_n,no_default_updates=True)
nearly_zeros=function(inputs=[],outputs=rv_u+rv_u-2*rv_u)
print f()
print f()
print g()
print g()
print nearly_zeros()
