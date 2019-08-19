""" Deep neural network
    L : 网络的层数
    N_L : 第L层的神经元数量， N_0 : 输入层
    a_l = g_l(z_l) : 第L层的激活函数， 特别的 a_l[0] = X, a_l[L] = y_hat
    w_l , b_l : 第L层激活函数的权重系数和偏置值

    如何判断每一层的参数维度是否正确？：
        w_l = (n_l, n_l-1) -> dw_l
        b_l = (n_l, 1) -> db_l

    问题：1.参数初始化的问题： w，b 的不同初始化方式，会对结果产生非常大的影响，主要是通过影响 学习率 的选择来发生的。
             :random初始化,最佳学习率在0.06附近
             :he 初始化，最佳学习率在0.001附近
             :zeros 初始化，不推荐使用，极容易发生梯度消失
"""
import h5py
import time
import numpy as np
import warnings
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings('ignore')


class DeepNeuralNetwork:

    def __init__(self, n_iters=500, hidden_layer_init=None, learning_rate_init=0.001, print_cost=True, init_type='he'):
        self.w_ = None
        self.b_ = None
        self.n_iters = n_iters
        self.hidden_layer_ = list(hidden_layer_init)
        self.alpha = learning_rate_init
        self.print_cost = print_cost

    def _init_coef(self):
        assert self.hidden_layer_ is not None, "please input a hidden_layer_init!"
        L = len(self.hidden_layer_)
        w = [None]
        b = [None]

        # w[0], b[0]为空，暂时不作处理
        for i in range(1, L):
            w.append(np.random.random((self.hidden_layer_[i], self.hidden_layer_[i-1])) * 0.01)
            b.append(np.zeros((self.hidden_layer_[i], 1)))

        return np.array(w), np.array(b)

    def _sigmoid(self, z):
        return 1./(1+np.exp(-z))

    def _sigmoid_backward(self, dA, z):
        A = self._sigmoid(z)
        return dA * A * (1-A)

    def _one_forward(self, X, w, b):
        """ 实现一次前向传播计算
        :param X:
        :param y:
        :param w:
        :param b:
        :return: A_L, caches = [Z]
        """
        # 网络长度
        L = len(w)-1
        caches = [(None, X)]
        A = X

        # (L-1)层为隐藏层； L层为输出层，另外计算;w[0],b[0]为空
        for i in range(1, L):
            # input a[l-1]，即输入为上一层的计算结果
            A_pre = A
            # output a[l], z[l]，输出为当前层的计算结果
            Z = np.dot(w[i], A_pre) + b[i]
            A = self._sigmoid(Z)
            # 缓存
            caches.append((Z, A))

        # 单独计算L输出层
        Z = np.dot(w[-1], A) + b[-1]
        A_L = self._sigmoid(Z)
        caches.append((Z, A_L))

        return A_L, caches

    def _one_backward(self, A_L, y, forward_caches, w, b):
        """ 一次反向传播过程
        :param A_L:
        :param y:
        :param forward_caches: 每一层的(Z, A), 长度为输入的层数
        :return:
        """
        dA = []
        dW = []
        db = []
        # 网络层数, L层另外计算
        L = len(forward_caches)-1
        m = y.shape[1]
        y = y.reshape(A_L.shape)  # 确保两者规模相同
        # 成本函数[交叉熵]的关于AL的导数
        dZ_L = -(np.divide(y, A_L) - np.divide(1 - y, 1 - A_L))

        # 输出层单独计算
        dA.append(np.dot(w[L].T, dZ_L))
        dW.append(1./m * np.dot(dZ_L, forward_caches[L-1][1].T))
        db.append(1./m * np.sum(dZ_L, axis=1, keepdims=True))

        # 计算隐藏层, 注意forward_caches每层代表的层数，重点！
        for i in reversed(range(L-1)):
            dZ = self._sigmoid_backward(dA[0], forward_caches[i+1][0])
            dA.insert(0, np.dot(w[i+1].T, dZ))
            dW.insert(0, 1./m * np.dot(dZ, forward_caches[i][1].T))
            db.insert(0, 1./m * np.sum(dZ, axis=1, keepdims=True))

        return np.array(dW), np.array(db)

    def fit(self, X, y):
        n_features, m = X.shape

        w, b = self._init_coef()

        for i in range(self.n_iters):
            # 前向传播
            A_L, forward_caches = self._one_forward(X, w, b)
            # 反向传播
            dW, db = self._one_backward(A_L, y, forward_caches, w, b)
            # 更新参数
            for j in range(len(dW)):
                # print('w:',w[j+1].shape)
                # print('dW',dW[j].shape)
                w[j+1] -= self.alpha * dW[j]
                b[j+1] -= self.alpha * db[j]
            # 计算损失函数
            cost = -1 / m * (np.dot(y, np.log(A_L).T) + np.dot(1 - y, np.log(1 - A_L).T))
            cost = np.squeeze(cost)  # 确保cost是一个实数;标量

            if self.print_cost and i%100 == 0:
                print("Cost after iteration {}:{}".format(i, cost))

        self.w_ = w
        self.b_ = b

        return self

    def predict_proba(self, X):
        """ 预测结果概率矩阵
        :param X:
        :return: 概率矩阵
        """
        assert self.w_ is not None and self.b_ is not None, "must fit before predict!"
        proba, _ = self._one_forward(X, self.w_, self.b_)

        return proba

    def predict(self, X):
        """ 二分类，预测矩阵
        :param X:
        :param y:
        :return:
        """
        assert self.w_ is not None and self.b_ is not None, "must fit before predict!"
        proba, _ = self._one_forward(X, self.w_, self.b_)

        return np.array(proba >= 0.5, dtype='int')

    def score(self, X, y):
        assert self.w_ is not None and self.b_ is not None, "must fit before predict!"
        y_predict = self.predict(X)

        return accuracy_score(y_test.ravel(), y_predict.ravel())


if __name__ == '__main__':
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

        time0 = time.time()

        dnn = DeepNeuralNetwork(hidden_layer_init=(12288, 100, 20, 1), n_iters=500, learning_rate_init=0.06)
        dnn.fit(X_train, y_train)
        print('预测结果概率矩阵:', dnn.predict_proba(X_test))
        print('预测结果：', dnn.predict(X_test))
        print('模型准确率', dnn.score(X_test, y_test))
        time1 = time.time()
        print('cost time:', time1 - time0)
        """
        print("\n-------sklearn中的效果------")
        X_train = X_train.T
        X_test = X_test.T
        y_train = y_train.ravel()
        y_test = y_test.ravel()
        mlp = MLPClassifier(solver='sgd', activation='relu', alpha=1e-4, hidden_layer_sizes=(100, 100, 20))
        mlp.fit(X_train, y_train)
        print("loss", mlp.loss_)
        print("模型准确率：", mlp.score(X_test, y_test))
        """
