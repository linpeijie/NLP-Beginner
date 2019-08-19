import unittest
import time
import numpy as np
from neural_network.simple_nn import SimpleNeuralNetwork
from neural_network.LNN import LNNModle
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


class SNNTest(unittest.TestCase):

    def test_beginner(self):
        """ 简单的数据测试一下双层神经网络是否正常运行
        :return:
        """

        X_train = np.array([[0, 0, 1],
                            [1, 1, 1],
                            [1, 0, 1],
                            [0, 1, 1]]).T
        # 训练数据真实结果
        y_train = np.array([[0, 1, 1, 0]])
        print(X_train.shape, y_train.shape)

        X_test = np.array([[1, 0, 0]]).T
        y_test = np.array([[1]])

        ss = StandardScaler()
        X_train = ss.fit_transform(X_train)
        X_test = ss.fit_transform(X_test)

        time0 = time.time()

        snn = SimpleNeuralNetwork(activation='sigmoid', learning_rate_init=1)
        snn.fit(X_train, y_train)
        print('预测结果概率矩阵:', snn.predict_proba(X_test))
        print('预测结果：', snn.predict(X_test))
        time1 = time.time()
        print('cost time:', time1 - time0)

    def test_LNN(self):
        X_train = np.array([[0, 0, 1],
                            [1, 1, 1],
                            [1, 0, 1],
                            [0, 1, 1]]).T
        # 训练数据真实结果
        y_train = np.array([[0, 1, 1, 0]])
        print(X_train.shape, y_train.shape)

        X_test = np.array([[1, 0, 0]]).T
        y_test = np.array([[1]])

        ss = StandardScaler()
        X_train = ss.fit_transform(X_train)
        X_test = ss.fit_transform(X_test)

        layers_dims = [3, 4, 1]
        model = LNNModle()
        parameters, _ = model.model(X_train, y_train, layers_dims, num_iters=500, learning_rate=0.001, print_cost=True)
        results = model.score(parameters, X_test, y_test)
        print('---准确率----:', results)
