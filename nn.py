
from dataclasses import dataclass
from numpy import ndarray, random, exp, multiply, sqrt, zeros, linalg
from numpy.core.fromnumeric import argmax
from numpy.core.numeric import zeros_like

@dataclass
class layer:
  name: str
  w: ndarray
  b: ndarray

def sigmoid(a):
  return 1 / (1 + exp(-a))

def sigmoid_prime(a):
  return sigmoid(a) * (1 - sigmoid(a))

class nn:
  def __init__(self, K: int):    
    self.l1 = layer('1', random.normal(size=(784, K)), random.normal(size=(K, 1)))
    self.l2 = layer('2', random.normal(size=(K, 10)), random.normal(size=(10, 1)))
  
  def predict(self, X: ndarray):
    W1, b1 = self.l1.w, self.l1.b
    W2, b2 = self.l2.w, self.l2.b

    h1 = sigmoid(W1.T @ X  + b1)
    h2 = sigmoid(W2.T @ h1 + b2)

    return h2
  
  def predict_batch(self, X :ndarray):
    (N, _) = X.shape

    y = zeros(shape = (N, 10))

    for n in range(N):
      xn = X[n, :].reshape((784, 1))
      y[n] = self.predict(xn).T

    return y


  def grads(self, xn: ndarray, yn: ndarray):
    W1, b1 = self.l1.w, self.l1.b
    W2, b2 = self.l2.w, self.l2.b

    def forward():
      x0 = xn 
      z1 = W1.T @ x0 + b1
      x1 = sigmoid(z1)
      z2 = W2.T @ x1 + b2
      x2 = sigmoid(z2)
      return { 
        'xs': (x0, x1, x2),
        'zs' : (z1, z2),
      }

    def backward(xs, zs):
      x0, x1, x2 = xs
      z1, z2 = zs

      d2 = multiply((x2 - yn), sigmoid_prime(z2))
      d1 = multiply(W2 @ d2, sigmoid_prime(z1))

      dldw2 = x1 @ d2.T
      dldb2 = d2
      dldw1 = x0 @ d1.T
      dldb1 = d1

      return (dldw2, dldb2, dldw1, dldb1)

    fwd = forward()
    xs = fwd['xs']
    zs = fwd['zs']
    return backward(xs, zs)

  def loss(self, X, y):
    f = self.predict_batch(X)
    return 0.5 * linalg.norm((f - y))
  
  def accuracy(self, X, y):
    f = self.predict_batch(X)

    (N, _) = X.shape

    correct = 0
    total = 0

    for n in range(N):
      fn = argmax(f[n, :])
      yn = argmax(y[n, :])
      if fn == yn:
        correct += 1
      total += 1
    
    return (correct / total)


  def train(self, epochs, X, y, alpha, batch_size=64):
    (N, _) = X.shape
    B = batch_size

    for epoch in range(epochs):
      if epoch % 100 == 0:
        print('epoch: ' + str(epoch))

      W1, b1 = self.l1.w, self.l1.b
      W2, b2 = self.l2.w, self.l2.b

      dldw2 = zeros_like(W2)
      dldb2 = zeros_like(b2)
      dldw1 = zeros_like(W1)
      dldb1 = zeros_like(b1)

      for _ in range(B):
        n = random.randint(low=0, high=N-1)
        xn = X[n, :].reshape((784, 1))
        yn = y[n, :].reshape((10, 1))
        (dldw2pr, dldb2pr, dldw1pr, dldb1pr) = self.grads(xn, yn)
        dldw2 += dldw2pr
        dldb2 += dldb2pr
        dldw1 += dldw1pr
        dldb1 += dldb1pr
      
      dldw2 = (1 / B) * dldw2
      dldb2 = (1 / B) * dldb2
      dldw1 = (1 / B) * dldw1
      dldb1 = (1 / B) * dldb1

      self.l2.w -= alpha * dldw2
      self.l2.b -= alpha * dldb2
      self.l1.w -= alpha * dldw1
      self.l1.b -= alpha * dldb1