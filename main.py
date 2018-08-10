import numpy
import random
"""
dataZ1 = [7, 3, 0]
dataZ2 = [5, 2, 0]
dataZ3 = [2, 6, 0]
dataZ4 = [10, 5, 0]

dataO1 = [110, 60, 1]
dataO2 = [150, 80, 1]
dataO3 = [90, 70, 1]
dataO4 = [170, 100, 1]"""

#allxd = [dataZ1, dataZ2, dataZ3, dataZ4, dataO1, dataO2, dataO3, dataO4]
allxd = [[3,   1.5, 1],
        [2,   1,   0],
        [4,   1.5, 1],
        [3,   1,   0],
        [3.5, .5,  1],
        [-2,   -.5,  0],
        [5.5,  1,  1],
        [-1,    -1,  0]]

random.seed(12)

w1 = random.random()
w2 = random.random()
b = random.random()

print("w1: {}, w2: {}, b: {}".format(w1, w2, b))

def NN(m1, m2):
    z = m1*w1 + m2*w2 + b
    return sigmoid(z)

def _NN(m1, m2):
    return m1*w1 + m2*w2 + b
    
def sigmoid(x):
    return 1/(1 + numpy.exp(-x))

def dsigmoid(x):
    return sigmoid(x)*(1 - sigmoid(x))

def error(m1,m2,result):
    return (NN(m1,m2) - result)**2

def cost(lista):
    costo = 0
    for m in lista:
        costo += error(m[0], m[1], m[2])
    return costo

def _dedpred(m1,m2,res): 
    return 2*(NN(m1,m2) - res)

def _dedw1(m1, m2, res):
    return _dedpred(m1, m2, res)*dsigmoid(_NN(m1,m2))*m1

def _dedw2(m1, m2, res):
    return _dedpred(m1, m2, res)*dsigmoid(_NN(m1,m2))*m2

def _dedb(m1, m2, res):
    return _dedpred(m1, m2, res)*dsigmoid(_NN(m1,m2))

def dc(lista):
    dw1 = 0
    dw2 = 0
    db = 0
    for m in lista:
        dw1 += _dedw1(m[0], m[1], m[2])
        dw2 += _dedw2(m[0], m[1], m[2])
        db += _dedb(m[0], m[1], m[2])
    return (dw1, dw2, db)

# training
for i in range(50000):
    print("Iter: ", i, " - Costo: ", cost(allxd))
    gradiente = dc(allxd)
    w1 = w1 - .2*gradiente[0]
    w2 = w2 - .2*gradiente[1]
    b = b - .2*gradiente[2]

for a in allxd:
    print("NN en ", a,": ", NN(a[0], a[1]))

print("Prediccion en (5,7): ", NN(5,7))

print("w1: {}, w2: {}, b: {}".format(w1, w2, b))

