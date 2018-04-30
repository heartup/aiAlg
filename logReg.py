import numpy as np
import matplotlib.pyplot as plt


# def nonlin(x, deriv=False):
#     if deriv:
#         return 1. * (x > 0)
#     else:
#         return x * (x > 0)

def nonlin(x, deriv=False):
    if (deriv == True):
        return x * (1.0 - x)

    return 1.0 / (1.0 + np.exp(-x))


np.random.seed(0)
N = 100  # number of points per class
f = 14  # feature 数量
D = 2  # dimensionality
K = 3  # number of classes
X = np.zeros((N * K, f))
y = np.zeros((N * K, 3), dtype='uint8')
# used for feature scaling
avgX = np.zeros((1, f))
maxX = np.zeros((1, f))

for j in range(K):
    ix = range(N * j, N * (j + 1))
    r = np.linspace(0.0, 1, N)  # radius
    t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
    xx = r * np.sin(t)
    yy = r * np.cos(t)

    X[ix] = np.c_[xx,
                  yy,
                  np.power(xx, 2),
                  np.power(yy, 2),
                  np.power(xx, 3),
                  np.power(yy, 3),
                  np.power(xx, 4),
                  np.power(yy, 4),
                  xx * yy,
                  np.power(xx, 2) * yy,
                  xx * np.power(yy, 2),
                  np.power(xx, 3) * yy,
                  np.power(yy, 3) * xx,
                  np.power(xx, 2) * np.power(xx, 2)]

    y[ix] = np.c_[j == 0, j == 1, j == 2]

avgX = X.mean(0)
maxX = X.max(0)

X = (X - avgX) / maxX

h = 100  # size of hidden layer
l = 3  # num of layer

W = [2 * np.random.randn(f, h) - 1]

for i in range(l - 3):
    W.append(2 * np.random.randn(h, h) - 1)

W.append(2 * np.random.randn(h, 3) - 1)

step_size = 0.0001
reg_step = 0.00000001

num_examples = X.shape[0]
for i in range(10050):
    a = [X]
    for j in range(l - 1):
        a.append(nonlin(np.dot(a[j], W[j])))

    d_y = a[l - 1] - y

    err = np.mean(np.abs(d_y))
    if (i % 100) == 0:
        print("Error:" + str(err))

    d_z = [nonlin(d_y, deriv=True)]
    for j in range(l - 2):
        d_z.insert(0, nonlin(d_z[0].dot(W[l - j - 2].T), deriv=True))

    for j in range(l - 1):
        W[j] -= step_size * a[j].T.dot(d_z[j])
        W[j] -= reg_step * W[j]

a = [X]
for j in range(l - 1):
    a.append(nonlin(np.dot(a[j], W[j])))

d_y = a[l - 1] - y
# if (i % 100) == 0:
print("Final Error:" + str(np.mean(np.abs(d_y))))

h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

xxr = xx.ravel()
yyr = yy.ravel()

a0 = np.c_[xxr,
          yyr,
          np.power(xxr, 2),
          np.power(yyr, 2),
          np.power(xxr, 3),
          np.power(yyr, 3),
          np.power(xxr, 4),
          np.power(yyr, 4),
          xxr * yyr,
          np.power(xxr, 2) * yyr,
          xxr * np.power(yyr, 2),
          np.power(xxr, 3) * yyr,
          np.power(yyr, 3) * xxr,
          np.power(xxr, 2) * np.power(xxr, 2)]

a0 = (a0 - avgX) / maxX

a = [a0]

for j in range(l - 1):
    a.append(nonlin(np.dot(a[j], W[j])))

Z = a[l - 1]
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)
fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=np.argmax(y, axis=1), s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()
