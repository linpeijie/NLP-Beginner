import numpy as np
import matplotlib.pyplot as plt


class Perceptron:

    def __init__(self, w0, b0):
        self.W_ = w0
        self.b_ = b0
        self.n_ = 1

    def loss_func(self, x, y):
        return y * (self.W_.dot(np.array(x).T) + self.b_)

    def fit(self, X_train, y_train):

        fine = False
        print('---误分类点------- w ---------b')

        while not fine:
            fine = True

            for x, y in zip(X_train, y_train):
                if self.loss_func(x, y) <= 0:
                    self.W_ = self.W_ + self.n_ * np.dot(y, x)
                    self.b_ = self.b_ + self.n_ * y
                    fine = False
                    print('   ', x, '     ', self.W_, '     ', self.b_)

        return self.W_, self.b_


if __name__ == '__main__':

    X = np.array([[3, 3],
                 [4, 3],
                 [1, 1]])

    Y = np.array([1, 1, -1])

    W0 = np.array([0, 0])
    b0 = 0

    per = Perceptron(W0, b0)
    w, b = per.fit(X, Y)

    def x2(x1, w, b):
        return (w[0] * x1 + b) / -w[1]

    x1 = np.linspace(0, 6, 100)
    x2 = x2(x1, w, b)

    plt.scatter(X[:, 0], X[:, 1], color='red')
    plt.xlim((0, 6))
    plt.ylim((0, 6))
    plt.plot(x1, x2)
    plt.show()
