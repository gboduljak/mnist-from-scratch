import numpy as np
from load_mnist import load_mnist


def one_hot(y_train):
  (N, _) = y_train.shape
  y = np.zeros(shape=(N, 10))

  for k in range(N):
    y[k][y_train[k]] = 1
  
  return y

np.random.seed(42)

def get_dataset():

  images, labels = load_mnist(path="./dataset", section="training")

  X_train = images.reshape((60000, 784))
  y_train = labels.reshape((60000, 1))

  images, labels = load_mnist(path="./dataset", section="testing")
  X_val = images.reshape((10000, 784))
  y_val = labels.reshape((10000, 1))

  y_train = one_hot(y_train)
  y_val = one_hot(y_val)

  return (X_train, y_train), (X_val, y_val)