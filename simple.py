import numpy as np

RANDOM_SEED = 123
randomNumberGenerator = np.random.RandomState(RANDOM_SEED)

d = 2   # dimension
N = 10  # number of data
mean = 5

x1 = randomNumberGenerator.randn(N, d) + np.array([0, 0])
x2 = randomNumberGenerator.randn(N, d) + np.array([mean, mean])

x = np.concatenate((x1, x2), axis=0)

w = np.zeros(d) # weight
b = 0           # bias

def y(x):
    return step(np.dot(w, x) + b)

def step(x):
    return 1 * (x > 0)

def t(i):
    # return 1 * (i >= N)
    if i < N:
        return 0
    else:
        return 1


# Error Correction Method
while True:
    classified = True
    for i in range(N * 2):
        delta_w = (t(i) - y(x[i])) * x[i]
        delta_b = (t(i) - y(x[i]))
        w += delta_w
        b += delta_b
        classified *= all(delta_w == 0) * (delta_b == 0)
    if classified:
        break

print(w, b)