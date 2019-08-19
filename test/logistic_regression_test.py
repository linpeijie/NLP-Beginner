import unittest
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from LR.logistic_regression import LogisticRegression
import matplotlib.pyplot as plt


class TestLogisticRegression(unittest.TestCase):

    def test_iris_multi_feature(self):
        """使用更多特征来做分类
           正确性来自于多元线性回归的性质，不管特征有多少，最终结果是求出一个回归值（一个数），
           而罗辑回归模型是根据该回归值来计算该 样本 在全体正态分布上的概率值
           最终通过 sigmoid 函数判别
        """
        iris = datasets.load_iris()

        X = iris.data
        y = iris.target
        X = X[y < 2, :]
        y = y[y < 2]

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
        log_reg = LogisticRegression()
        log_reg.fit(X_train, y_train)

        print('---预测准确率---\n', log_reg.score(X_test, y_test))
        print('---预测结果---\n', log_reg.predict(X_test))

    def test_iris(self):
        """使用两个特征来做分类"""
        iris = datasets.load_iris()
        pre_res = [1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0]

        X = iris.data
        y = iris.target
        X = X[y < 2, : 2]
        y = y[y < 2]

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
        log_reg = LogisticRegression()
        log_reg.fit(X_train, y_train)

        self.assertTrue(isinstance(log_reg, LogisticRegression), msg='罗辑回归类调用失败')
        self.assertEqual(log_reg.score(X_test, y_test), 1.0, msg='预测结果不正确')
        self.assertListEqual(list(log_reg.predict(X_test)), pre_res, msg='预测结果不正确')

        def x2(x1):
            return (-log_reg.coef_[0] * x1 - log_reg.interception_) / log_reg.coef_[1]

        x1_plot = np.linspace(4, 8, 1000)
        x2_plot = x2(x1_plot)

        plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red')
        plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue')
        plt.plot(x1_plot, x2_plot)
        plt.show()

    def test_multi(self):
        """多项式逻辑回归，即将 多元线性回归方程式 换成 非线性回归方程式 ，即将 直线 换成 曲线 ，
           以适应更复杂的二分类问题
        """
        np.random.seed(666)
        X = np.random.normal(0, 1, size=(200, 2))
        y = np.array(X[:, 0]**2 + X[:, 1]**2 < 1.5, dtype='int')

        log_reg = LogisticRegression()
        log_reg.fit(X, y)

        self.plot_decision_boundary(log_reg, axis=[-4, 4, -4, 4])
        plt.scatter(X[y == 0, 0], X[y == 0, 1])
        plt.scatter(X[y == 1, 0], X[y == 1, 1])
        plt.show()

        poly_log_reg = self.PolynomialLogisticRegression(degree=2)
        poly_log_reg.fit(X, y)

        self.plot_decision_boundary(poly_log_reg, axis=[-4, 4, -4, 4])
        plt.scatter(X[y == 0, 0], X[y == 0, 1])
        plt.scatter(X[y == 1, 0], X[y == 1, 1])
        plt.show()

    def plot_decision_boundary(self, model, axis):
        """决策边界绘制"""

        x0, x1 = np.meshgrid(
            np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(-1, 1),
            np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1, 1)
        )

        # 按列合并多个向量
        X_new = np.c_[x0.ravel(), x1.ravel()]

        y_predict = model.predict(X_new)
        zz = y_predict.reshape(x0.shape)

        from matplotlib.colors import ListedColormap
        custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])

        plt.contourf(x0, x1, zz, linewidth=5, cmap=custom_cmap)

    def PolynomialLogisticRegression(self, degree):
        """使用管道（Pipeline）对特征添加多项式项
           degree 代表多项式的最高次数
           换句话说，就是把 wx直线，换成更复杂的 ax^2 + ax^2 曲线
        """
        return Pipeline([
            # 管道第一步：给样本特征添加多形式项；
            ('poly', PolynomialFeatures(degree=degree)),
            # 管道第二步：数据归一化处理；
            ('std_scaler', StandardScaler()),
            ('log_reg', LogisticRegression())
        ])


if __name__ == '__main__':
    unittest.main()
