import copy
import numpy as np

def sigmoid(x):
	return 1/(1+np.exp(-x))

def sigmoid_grad(x):
	return x*(1-x)

