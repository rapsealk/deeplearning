#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np

RANDOM_SEED = 123

randomNumberGenerator = np.random.RandomState(RANDOM_SEED)

dimension = 2
N = 10
mean = 5

x1 = randomNumberGenerator.randn(N, dimension) + np.array([0, 0])
x2 = randomNumberGenerator.randn(N, dimension) + np.array([mean, mean])

x = np.concatenate((x1, x2), axis=0)

w = np.zeros(dimension)
b = 0

def y(x):
    return step(np.dot(w, x) + b)

def step(x):
    return 1 * (x > 0)

def t(i):
    return 0 if i < N else 1

while True:
    classified = True
    for i in range(N * 2):
        delta_w = (t(i) - y(x[i])) * x[i]
        delta_b = (t(i) - y(x[i]))
        x += delta_w
        b += delta_b
        classified *= all(delta_w == 0) * (delta_b == 0)
    if classified: break

print(y([0, 0]))
print(y([5, 5]))