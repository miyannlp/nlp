import torch
from torch import tensor

def matmul(a, b):
    ar, ac = a.shape
    br, bc = b.shape
    assert ac == br
    c = torch.zeros(ar, bc)
    for i in range(ar):
        for j in range(bc):
            for k in range(ac):
                c[i, j] += a[i, k] * b[k, j]
    return c

m1 = torch.randn(5, 28*28)
m2 = torch.randn(784, 10)

t1 = matmul(m1, m2)

print(t1.shape)

a = tensor([10., 6, -4])
b = tensor([2.0, 8, 7])
print (a + b)

# Reduction operations like all, sum, and mean return tensors with only one element, called rank-0 tensors. If you want to convert this to a plain Python Boolean or number, you need to call .item:
y = (a + b).mean().item()
print(y)

# To access one column or row, we can simply write a[i,:] or b[:,j]. The : means take everything in that dimension. We could restrict this and take only a slice of that dimension by passing a range, like 1:5

# elementwise arithmetic and broadcasting

# Broadcasting with a scalar
a = tensor([10., 6, -4])
print(a > 0)

m = tensor([[1.0, 2, 3], [4, 5, 6], [7, 8, 9]])
print((m-5)/2.73)

# Broadcasting a vector to a matrix
c = tensor([10., 20, 30])
m = tensor([[1., 2, 3], [4, 5, 6], [7, 8, 9]])
print(m.shape, c.shape)

print (m+c)

# This is done by the expand_as method behind the scenes
print (c.expand_as(m))

# If we look at the corresponding tensor, we can ask for its storage property (which shows the actual contents of the memory used for the tensor) to check there is no useless data stored
t = c.expand_as(m)
print(t.storage())

# In fact, it’s only possible to broadcast a vector of size n with a matrix of size m by n:

# backward pass: compute all the gradients of a given loss with respect to its parameters, which is known as the backward pass
# forward pass: compute the output of the model on a given input
# so initializing the weights properly is extremely important.
# Xavier initialization (or Glorot initialization)
from math import sqrt
w1 = torch.randn(100, 50) / sqrt(100)
b1 = torch.zeros(50)
w2 = torch.randn(50, 1) / sqrt(50)
b2 = torch.zeros(1)

def lin(x, w, b): return x @ w + b

x = torch.randn(200, 100) # inputs are 200 vectors of size 100
y = torch.randn(200)  # targets are 200 random floats

l1 = lin(x, w1, b1)
print(l1.mean(), l1.std())

def relu(x): return x.clamp_min(0.)

l2 = relu(l1)
print(l2.mean(), l2.std())

# we should use the following scale instead: sqrt(2/nin), where nin is the number 
w1 = torch.randn(100, 50) * sqrt(2 / 100)
b1 = torch.zeros(50)
w2 = torch.randn(50, 1) * sqrt(2 / 50)
b2 = torch.zeros(1)

l1 = lin(x, w1, b1)
l2 = relu(l1)
print(l2.mean(), l2.std())

def model(x):
    l1 = lin(x, w1, b1)
    l2 = relu(l1)
    l3 = lin(l2, w2, b2)
    return l3

out = model(x)
print(out.shape)

# To get rid of this trailing 1 dimension, we use the squeeze function:
def mse(output, targ):
    return (output.squeeze(-1) - targ).pow(2).mean()

loss = mse(out, y)

print(loss)

# gradients and the backward pass

def mse_grad(inp, targ):
    # grad of loss with respect to output of previous layer
    # The derivative of the mean is just 1/n, where n is the number of elements in our input
    inp.g = 2. * (inp.squeeze() - targ).unsqueeze(-1) / inp.shape[0]

def relu_grad(inp, out):
    # grad of relu with respect to input activations
    inp.g = (inp > 0).float() * out.g

def lin_grad(inp, out, w, b):
    # grad of matmul with respect to input
    inp.g = out.g @ w.t()
    w.g = inp.t() @ out.g
    b.g = out.g.sum(0)

# t() Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.
# 0-D and 1-D tensors are returned as is. 
a = torch.randn(2, 3)
print(a)
print(a.t())

def forward_and_backward(inp, targ):
    # forward pass:
    l1 = inp @ w1 + b1
    l2 = relu(l1)
    out = l2 @ w2 + b2
    # we don't actually need the loss in backward!
    loss = mse(out, targ)

    # backward pass:
    mse_grad(out, targ)
    lin_grad(l2, out, w2, b2)
    relu_grad(l1, l2)
    lin_grad(inp, l1, w1, b1)

forward_and_backward(x, y)

# refactoring the model
class Relu():
    def __call__(self, inp):
        self.inp = inp
        self.out = inp.clamp_min(0.)
        return self.out
    
    def backward(self):
        self.inp.g = (self.inp > 0).float() * self.out.g
    
class Lin():
    def __init__(self, w, b):
        self.w, self.b = w, b
    
    def __call__(self, inp):
        self.inp = inp
        self.out = inp @ self.w + self.b
        return self.out
    
    def backward(self):
        self.inp.g = self.out.g @ self.w.t()
        self.w.g = self.inp.t() @ self.out.g
        self.b.g = self.out.g.sum(0)
    
class Mse():
    def __call__(self, inp, targ):
        self.inp = inp
        self.targ = targ
        self.out = (inp.squeeze() - targ).pow(2).mean()
        return self.out
    
    def backward(self):
        x = (self.inp.squeeze()-self.targ).unsqueeze(-1)
        self.inp.g = 2.*x/self.targ.shape[0]
    
class Model():
    def __init__(self, w1, b1, w2, b2):
        self.layers = [Lin(w1, b1), Relu(), Lin(w2, b2)]
        self.loss = Mse()

    def __call__(self, x, targ):
        for l in self.layers: x = l(x)
        return self.loss(x, targ)
    
    def backward(self):
        self.loss.backward()
        for l in reversed(self.layers): l.backward()

# instantiate our model
model = Model(w1, b1, w2, b2)

loss = model(x, y)
print(loss)

model.backward()

# going to PyTorch
# Lin, Mse and Relu classes we wrote have a lot in common, so we could make them all inherit from the same base class
class LayerFunction():
    def __call__(self, *args):
        self.args = args
        self.out = self.forward(*args)
        return self.out
    
    def forward(self):
        raise Exception('not implemented')
    
    def bwd(self):
        raise Exception('not implemented')

    def backward(self):
        self.bwd(self.out, *self.args)

class Relu(LayerFunction):
    def forward(self, inp):
        return inp.clamp_min(0.)
    
    def bwd(self, out, inp):
        inp.g = (inp>0).float() * out.g

class Lin(LayerFunction):
    def __init__(self, w, b):
        self.w, self.b = w, b
    
    def forward(self, inp):
        return inp @ self.w + self.b
    
    def bwd(self, out, inp):
        inp.g = out.g @ self.w.t()
        self.w.g = inp.t() @ self.out.g
        self.b.g = out.g.sum(0)

class Mse(LayerFunction):
    def forward(self, inp, targ):
        return (inp.squeeze() - targ).pow(2).mean()
    
    def bwd(self, out, inp, targ):
        inp.g = 2 * (inp.squeeze()-targ).unsqueeze(-1)/targ.shape[0]

# instantiate our model
model = Model(w1, b1, w2, b2)

loss = model(x, y)
print(loss)

model.backward()


# to implement an nn.Module you just need to do the following:
# 1. make sure the superclass __init__ is called first when you initialize it
# 2. define any parameters of the model as attributes with nn.Parameter
# 3. define a forward function that returns the output of your model

import torch.nn as nn
class LinearLayer(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_out, n_in) * sqrt(2/n_in))
        self.bias = nn.Parameter(torch.zeros(n_out))
    
    def forward(self, x):
        return x @ self.weight.t() + self.bias
    
lin = LinearLayer(10, 2)
p1, p2 = lin.parameters()
print(p1.shape, p2.shape)

class Model(nn.Module):
    def __init__(self, n_in, nh, n_out):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(n_in, nh), nn.ReLU(), nn.Linear(nh, n_out))
        self.loss = mse
    
    def forward(self, x, targ):
        return self.loss(self.layers(x).squeeze(), targ)
    

    
    

