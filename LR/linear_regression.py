import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class SimpleLinearRegression:

    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        """根据训练数据集x_train, y_train训练模型"""
        assert x_train.ndim == 1, "Simple Linear Regression can only solve single feature training datasets."
        assert len(x_train) == len(y_train), "the size of x_train must be equal to the size of y_train"

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        numerator = (x_train - x_mean).dot(y_train - y_mean)  # .dot 是 dot product 求内积
        denominator = (x_train - x_mean).dot(x_train - x_mean)

        self.a_ = numerator / denominator
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        """给定待预测数据集x_predict，返回表示x_predict的结果向量"""
        assert x_predict.ndim == 1, "Simple Linear Regression can only solve single feature training datasets."
        assert self.a_ is not None and self.b_ is not None, "must fit before predict!"

        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        """给定单个待预测数据x_single， 返回x_single的预测结果值"""
        return self.a_ * x_single + self.b_

    def __repr__(self):
        return "SimpleLinearRegression()"


class LinearRegression:

    def __init__(self):
        """"""
        self.coef_ = None
        self.interception_ = None
        self._theta = None

    def fit_normal(self, X_train, y_train):
        """训练模型"""
        assert X_train.shape[0] == y_train.shape[0], "the size of X_train must be equal to the size of y_train."

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)  # linalg.inv求逆

        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def fit_gd(self, X_train, y_train, eta=0.01, n_iters=1e4):
        """用梯度下降法训练模型,即求损失函数最小时的参数theta
           若数据量特别大，且数据数值维度跨度比较大，则需要先进行数据归一化处理
           梯度下降法的优势在于特征规模特别大时，能够极大的缩短模型训练时间
        """
        assert X_train.shape[0] == y_train.shape[0], "the size of X_train must be equal to the size of y_train"

        def J(theta, X_b, y):
            """损失函数"""
            try:
                return np.sum((y - X_b.dot(theta)) ** 2) / len(y)
            except:
                return float('inf')

        def dJ(theta, X_b, y):
            """求解每个theta的偏导数"""
            # res = np.empty(len(theta))
            # res[0] = np.sum(X_b.dot(theta) - y)
            # for i in range(1, len(theta)):
            #     res[i] = (X_b.dot(theta) - y).dot(X_b[:, i])
            # return res * 2 / len(X_b)
            return X_b.T.dot(X_b.dot(theta) - y) * 2 / len(X_b)  # 向量化运算

        def gradient_descent(X_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):
            theta = initial_theta
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient  # 梯度下降，每次前进eta距离
                if abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon:
                    break

                cur_iter += 1

            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])

        self._theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters)

        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def predict(self, X_predict):
        """预测,返回结果向量"""
        assert self.interception_ is not None and self.coef_ is not None, "must fit before predict!"
        assert X_predict.shape[1] == len(self.coef_), "the feature number of X_predict must be equal to X_train."

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta)

    def score(self, X_test, y_test):
        """模型准确度"""
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "LinearRegression("


if __name__ == '__main__':
    boston = datasets.load_boston()

    X = boston.data
    y = boston.target

    X = X[y < 50.0]
    y = y[y < 50.0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

    reg1 = LinearRegression()
    reg1.fit_normal(X_train, y_train)
    print("fit normal score:", reg1.score(X_test, y_test))

    reg2 = LinearRegression()
    # 对数据进行归一化
    standardScaler = StandardScaler()
    standardScaler.fit_transform(X_train)
    X_train_standard = standardScaler.transform(X_train)
    X_test_standard = standardScaler.transform(X_test)

    reg2.fit_gd(X_train_standard, y_train)
    print("fit gd score:", reg2.score(X_test_standard, y_test))
