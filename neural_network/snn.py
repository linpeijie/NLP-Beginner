"""
    遇到的问题：
    1.如果用self.w -= self.learning_rate_init * dw , 来更新参数，会发生self.w 数值不改变的问题？？？为什么？？？
"""
import h5py
import numpy as np
import json
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')


class SimpleNeuralNetword:

    def __init__(self, n_iters=500, learning_rate_init=0.001, print_cost=True):
        self.w = None
        self.b = None
        self.n_iters = n_iters
        self.learning_rate_init = learning_rate_init
        self.print_cost = print_cost

    def _sigmoid(self, z):
        """
        :param z:
        :return:
        """
        return 1.0/(1+np.exp(-z))

    def _init_coef(self, n_features):
        """ 初始化参数，单层神经网络，只有一个输出层
        :param n_features:
        :return:
        """
        w = np.random.random((1, n_features)) * 0.01
        b = 0

        return w, b

    def propagate(self, w, b, X, y):
        """ 单层神经网络的前向传播和反向传播
        :param X:
        :param y:
        :return:
        """
        m = X.shape[1]

        # 前向传播
        Z = np.dot(w, X) + b
        A = self._sigmoid(Z)
        cost = - 1/m * np.sum(y*np.log(A) + (1-y)*np.log(1-A))

        # 反向传播
        dz = A - y
        dw = 1/m * np.dot(dz, X.T)
        db = 1/m * np.sum(dz, axis=1, keepdims=True)

        return dw, db, cost

    def fit(self, X, y):
        """ 训练模型
        :param X:
        :param y:
        :param n_iters:
        :param learning_rate_init:
        :param print_cost:
        :return:
        """
        n_features, m = X.shape
        w, b = self._init_coef(n_features)

        for i in range(self.n_iters):
            dw, db, cost = self.propagate(w, b, X, y)
            w -= self.learning_rate_init * dw
            b -= self.learning_rate_init * db

            if self.print_cost and i % 100 == 0:
                print("Cost after iteration {}:{}".format(i, cost))

        self.w = w
        self.b = b

        return self

    def predict_proba(self, X):
        """ 预测结果概率矩阵
        :param X:
        :return:
        """
        m = X.shape[1]
        yHat = np.zeros((1, m))
        y_proba = self._sigmoid(np.dot(self.w, X) + self.b)  # 前向传播过程

        return y_proba

    def predict(self, X):
        """预测结果
        :param X:
        :return:
        """
        proba = self.predict_proba(X)
        return np.array(proba >= 0.5, dtype='int')

    def score(self, X_test, y_test):
        """模型准确度"""
        y_predict = self.predict(X_test)
        return accuracy_score(y_test.ravel(), y_predict.ravel())


if __name__ == '__main__':
    def load_datasets():
        """ 加载小猫咪数据集, X_train.shape(209, 64, 64, 3)表示这是一张64*64像素图片，包含有3原色信息(即3表示每个像素点有三种颜色信息)
        :return: X_tiran, y_train, X_test, y_test, class
        """
        train_dataset = h5py.File("../datasets/cat/train_catvnoncat.h5")
        train_x_orig = np.array(train_dataset['train_set_x'][:])
        train_y_orig = np.array(train_dataset['train_set_y'][:])

        test_dataset = h5py.File("../datasets/cat/test_catvnoncat.h5")
        test_x_orig = np.array(test_dataset['test_set_x'][:])
        test_y_orig = np.array(test_dataset['test_set_y'][:])

        classes = np.array(test_dataset['list_classes'][:])  # 类别信息

        train_y_orig = train_y_orig.reshape((1, train_y_orig.shape[0]))
        test_y_orig = test_y_orig.reshape((1, test_y_orig.shape[0]))

        # 4维数据2维化，方便计算
        X_train = train_x_orig.reshape(train_x_orig.shape[0], -1).T
        X_test = test_x_orig.reshape(test_x_orig.shape[0], -1).T

        ss = StandardScaler()
        X_train = ss.fit_transform(X_train)
        X_test = ss.fit_transform(X_test)

        return X_train, train_y_orig, X_test, test_y_orig, classes


    X_train, y_train, X_test, y_test, classes = load_datasets()

    print(X_train.shape, y_train.shape)
    snn = SimpleNeuralNetword()
    snn.fit(X_train, y_train)
    print("----预测准确率-----\n", snn.score(X_test, y_test))
