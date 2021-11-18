import numpy as np
from experiments import get_dataset
from nn import nn

np.random.seed(42)

train, val = get_dataset()

(X_train, y_train) = train
(X_val, y_val) = val

print('nn16 - 1024 epochs, batch=64')

model = nn(16)
model.train(1024, X_train, y_train, 10, batch_size=64)

print('Train loss: ' + str(model.loss(X_train, y_train)))
print('Train accuracy: ' + str(model.accuracy(X_train, y_train)))
print('Val accuracy: ' + str(model.accuracy(X_val, y_val)))
print('Val loss: ' + str(model.loss(X_val, y_val)))

print('nn16 - 2048 epochs, batch=64')

model = nn(16)
model.train(2048, X_train, y_train, 10, batch_size=64)

print('Train loss: ' + str(model.loss(X_train, y_train)))
print('Train accuracy: ' + str(model.accuracy(X_train, y_train)))
print('Val accuracy: ' + str(model.accuracy(X_val, y_val)))
print('Val loss: ' + str(model.loss(X_val, y_val)))

print('nn16 - 1024 epochs, batch=128')

model = nn(16)
model.train(1024, X_train, y_train, 10, batch_size=128)

print('Train loss: ' + str(model.loss(X_train, y_train)))
print('Train accuracy: ' + str(model.accuracy(X_train, y_train)))
print('Val accuracy: ' + str(model.accuracy(X_val, y_val)))
print('Val loss: ' + str(model.loss(X_val, y_val)))

print('nn16 - 2048 epochs, batch=128')

model = nn(16)
model.train(2048, X_train, y_train, 10, batch_size=128)

print('Train loss: ' + str(model.loss(X_train, y_train)))
print('Train accuracy: ' + str(model.accuracy(X_train, y_train)))
print('Val accuracy: ' + str(model.accuracy(X_val, y_val)))
print('Val loss: ' + str(model.loss(X_val, y_val)))


print('nn16 - 1024 epochs, batch=256')

model = nn(16)
model.train(1024, X_train, y_train, 10, batch_size=256)

print('Train loss: ' + str(model.loss(X_train, y_train)))
print('Train accuracy: ' + str(model.accuracy(X_train, y_train)))
print('Val accuracy: ' + str(model.accuracy(X_val, y_val)))
print('Val loss: ' + str(model.loss(X_val, y_val)))

print('nn16 - 2048 epochs, batch=256')

model = nn(16)
model.train(2048, X_train, y_train, 10, batch_size=256)

print('Train loss: ' + str(model.loss(X_train, y_train)))
print('Train accuracy: ' + str(model.accuracy(X_train, y_train)))
print('Val accuracy: ' + str(model.accuracy(X_val, y_val)))
print('Val loss: ' + str(model.loss(X_val, y_val)))
