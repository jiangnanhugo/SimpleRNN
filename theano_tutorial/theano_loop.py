import numpy as np
import  theano
import theano.tensor as T
'''
### Computing A^k
k=T.iscalar("k")
A=T.vector("A")

def _accumulate(x_tm1,A):
    return x_tm1*A

# symbolic description of the result
result,updates=theano.scan(
    fn=_accumulate,
    outputs_info=T.ones_like(A),
    non_sequences=A,
    n_steps=k
)


power=theano.function(
    inputs=[A,k],
    outputs=result[-1]
    #updates=updates
)
print (power(range(10),2))
print(power(range(10),4))

'''

coef=T.vector('coefficients')
x=T.scalar("x")

max_length=1000

def _calculate_coef(a,b,c):
    return a*(c**b)

# generate the components of the polynomial
components,_=theano.scan(
    fn=_calculate_coef,
    sequences=[coef,theano.tensor.arange(max_length)],
    non_sequences=x
)

calculate_polynomial=theano.function(
    inputs=[coef,x],
    outputs=components.sum()
)

coef=np.asarray([1,2,3],dtype=theano.config.floatX)
base=3
print(calculate_polynomial(coef,base))
print(1.0 * (3 ** 0) + 2. * (3 ** 1) + 3.0 * (3 ** 2))