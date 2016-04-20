import theano
import theano.tensor as T
import numpy as np


k = T.iscalar("k")
A = T.vector("A")
def inner_fct(prior_result, B):
    return B, B, prior_result * B   #!! change return of inner_fct
#!! change scan as below
[B1, B2, result], updates = theano.scan(fn=inner_fct,
                            outputs_info=[None, None, T.ones_like(A)],
                            non_sequences=A, n_steps=k)
final_result = result[-1]
power = theano.function(inputs=[A, k], outputs=final_result,
                      updates=updates)
print(power(range(10), 2))
'''
X=T.matrix('x')
W=T.matrix('w')
b=T.vector('b')

#tanh(x(t).dot(W) + b)
def _recurrence(v,W,b):
    return T.tanh(T.dot(v,W)+b)
result,updates=theano.scan(fn=_recurrence,
                           sequences=X,
                           non_sequences=(W,b))
compute_elementwise=theano.function(inputs=[X,W,b],
                                    outputs=result)

x=np.eye(2,dtype=theano.config.floatX)
w=np.ones((2,2),dtype=theano.config.floatX)
b=np.ones((2),dtype=theano.config.floatX)
b[1]=2

print(compute_elementwise(x,w,b))
print(np.tanh(x.dot(w)+b))

#x(t) = tanh(x(t - 1).dot(W) + y(t).dot(U) + p(T - t).dot(V))
X=T.vector("X")
Y=T.matrix("Y")
W=T.matrix('W')
b=T.vector('b')
U=T.matrix('U')
V=T.matrix('V')
P=T.matrix("P")

def _recurrence2(y,p,x_tm1):
    return T.tanh(T.dot(x_tm1,W)+T.dot(y,U)+T.dot(p,V))

results,_=theano.scan(fn=_recurrence2,
                        sequences=[Y,P[::-1]],
                        outputs_info=[X])
compute_seq=theano.function(inputs=[X,W,Y,U,P,V],
                            outputs=results)

x=np.zeros((2),dtype=theano.config.floatX)
x[1]=1
w=np.ones((2,2),dtype=theano.config.floatX)
y=np.ones((5,2),dtype=theano.config.floatX)
y[0,:]=-3
u=np.ones((2,2),dtype=theano.config.floatX)
p=np.ones((5,2),dtype=theano.config.floatX)
p[0,:]=3
v=np.ones((2,2),dtype=theano.config.floatX)

print(compute_seq(x,w,y,u,p,v))

# comparison with numpy
x_res=np.zeros((5,2),dtype=theano.config.floatX)
x_res[0]=np.tanh(x.dot(w)+y[0].dot(u)+p[4].dot(v))

for i in range(1,5):
    x_res[i]=np.tanh(x_res[i-1].dot(w)+y[i].dot(u)+p[4-i].dot(v))

print(x_res)


# Computing norms of lines of X
def _compute_norm(x):
    return T.sqrt((x**2).sum())
X=T.matrix("X")
results,_=theano.scan(fn=_compute_norm,
                      sequences=[X])
compute_norm_lines=theano.function(inputs=[X],
                                   outputs=results)

x=np.diag(np.arange(1,6,dtype=theano.config.floatX),1)
print x
print(compute_norm_lines(x))
print(np.sqrt(x**2).sum(1))
'''


# Computing norms of columns of X
X=T.matrix('X')
def _compute_norm_col(x):
    return T.sqrt((x**2).sum())
results,_=theano.scan(fn=_compute_norm_col(),
                      sequences=[X.T])
compute_norm_cols = theano.function(inputs=[X],
                                    outputs=results)
# test value
x = np.diag(np.arange(1, 6, dtype=theano.config.floatX), 1)
print(compute_norm_cols(x))

# comparison with numpy
print(np.sqrt((x ** 2).sum(0)))

'''

# x(t) = x(t - 2).dot(U) + x(t - 1).dot(V) + tanh(x(t - 1).dot(W) + b)

X=T.matrix("X")
W=T.matrix("W")
b=T.vector("b")
U=T.matrix('U')
V=T.matrix("V")
n_sym=T.iscalar('n_sym')

def _recurrence(x_tm2,x_tm1):
    return T.dot(x_tm2,U)+T.dot(x_tm1,V)+T.tanh(T.dot(x_tm1,W)+b)


results,_=theano.scan(
    fn=_recurrence,
    outputs_info=dict(initial=X,taps=[-2,-1]),
    n_steps=n_sym
)

compute_seq=theano.function(
    inputs=[X,U,V,W,b,n_sym],
    outputs=results
)

# the initial value must be able to return x[-2]
x=np.zeros((2,2),dtype=theano.config.floatX)
x[1,1]=1
w=0.5*np.ones((2,2),dtype=theano.config.floatX)
u=0.5*(np.ones((2,2),dtype=theano.config.floatX)-np.eye(2,dtype=theano.config.floatX))
v = 0.5 * np.ones((2, 2), dtype=theano.config.floatX)
n = 10
b = np.ones((2), dtype=theano.config.floatX)
print(compute_seq(x, u, v, w, b, n))
'''
