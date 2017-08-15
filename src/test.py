import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.array([[1, 1], [2, 2]])

np.random.uniform(low=0.1, high=2, size=(5, 6)) * 2 * 0.9 - 0.9

np.dot(a, b)

a = np.array([2, 3])
c = a.reshape((-1, 1))
b = np.array([[1, 1, 1, 1]])
print a.shape
print b
print c
print np.dot(c, b)
