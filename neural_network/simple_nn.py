""" 实现简单双层神经网络，包含一个input layer， 一个 hidden layer，一个output layer
    数据集采用公开的 猫咪图片数据集
    n_x 数据特征维度，m 训练样本总数, n_h 隐藏神经元数量, n_o=1 输出神经元数量
    Cost Function: J(w_1,b_1,w_2,b_2) = 1/m * np.sum(L(y_hat, y))
    Gradient Descent:
    repeat:
        1. compute prediction (y_hat_i, i=1,...,m)
        2. dw_1 = dJ/dw_1 , db_1 = dJ/db_1, so on ...
        3. w_1 -= alpha * dw_1  ;   b_1 = b_1 - alpha * db_2 ; so on ...
        4. cost = 1/m * -np.sum(y*np.log(y_hat) + (1-y)*np.log(1-y_hat))
    until:
        cost <= 某个数值
        或
        达到最大迭代次数 n_iters

    problems:
    1.梯度消失问题：不论是sigmoid，tanh，relu都会遇到梯度消失的问题，导致最终loss不再更新，所有的预测结果导向同一个值.
      即导数接近0，导致权重系数不再更新。
      但并不是每次都会发生梯度消失问题，为什么？
      目前发现的可能原因：
        1.数据标准化问题，最初的数据经过标准化处理后，loss相对应的减小了许多，但仍然会发生梯度消失问题，只不过消失的没有之前那样明显。
        2.之前发生的预测值全部相同的原因，也在于数据标准化问题。数据数值过大，梯度值过小，导致权重系数变化过小，因此计算出来的结果无限接近，最终表现为相同值
        3.与学习率无关
        4.发生梯度消失问题之后，所有的样本经过计算之后最终都会落入相近的区间里，导致最终计算结果十分相近。为什么？
        5.经过测试发现，是学习率过低导致的问题，这就引出另一个问题，在其他条件不变的情况下，是什么导致了学习率不同？
        6.经过测试，最终找到梯度消失问题：1.学习率过低，导致梯度消失。 2.w，b 参数初始化方式的不同，会通过影响学习率，而导致梯度消失问题。

"""
import h5py
import time
import numpy as np
import warnings
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings('ignore')


class SimpleNeuralNetwork:

    def __init__(self, hidden_layer_size=(4, 1), activation='sigmoid', learning_rate_init=0.001):
        """
        :w_1 : 隐藏层神经元权重系数，w_1.shape -> (n_h, n_x)
        :b_1 : 隐藏层神经元偏置值，b_1.shape -> (n_h, 1)
        :w_2 : 输出层神经元权重系数, w_2.shape -> (n_o, n_h)
        :b_2 : 输出神经元偏置值, b_2.shape -> (n_o, 1)
        :param hidden_layer_size: 隐藏层规模，行表示层数，列表示神经元个数
        """
        self.w_1 = None
        self.b_1 = None
        self.w_2 = None
        self.b_2 = None
        self.hidden_layer_size = hidden_layer_size
        self.activation = activation
        self.learning_rate_init = learning_rate_init

    def _sigmoid(self, z):
        """
        :param z: z为全部神经元线性方程计算出的结果，z.shape(1, n_h),z中每个元素代表一个神经元的输出
        :return: np.array().shape -> (1, n_h)
        """
        return 1. / (1.+np.exp(-z))

    def _sigmoid_backford(self, z):
        A = self._sigmoid(z)
        return A*(1-A)

    def _relu(self, z):
        """
        :param z: z为全部神经元线性方程计算出的结果，z.shape(1, n_h),z中每个元素代表一个神经元的输出
        :return: np.array().shape -> (1, n_h)
        """
        return np.maximum(np.zeros(z.shape), z)

    def _relu_backford(self, z):
        pass

    def _g(self, z, activation):
        """ 用来选择隐藏层的激活函数，由于只实现了二分类，所以输出层激活函数仍用sigmoid
        :param z: z为全部神经元线性方程计算出的结果，z.shape(1, n_h),z中每个元素代表一个神经元的输出
        :param activation: 激活函数
        :return: sigmoid or ReLU
        """
        if activation == 'sigmoid':
            return self._sigmoid(z)
        elif activation == 'relu':
            return self._relu(z)

    def _g_backford(self, z, activation):
        if activation == 'sigmoid':
            return self._sigmoid_backford(z)
        elif activation == 'relu':
            return self._ReLU(z)

    def propagate(self, X, Y, w_1, b_1, w_2, b_2):
        """ 实现 前向传播算法、反向传播算法
        :dz_2.shape -> (n_o, m)
        :dw_2.shape -> (n_o, n_h)
        :db_2.shape -> (n_o, 1)
        :dz_1.shape -> (n_h, m)
        :dw_1.shape = (n_h, n_x)
        :db_1.shape -> (n_h, 1)
        :param X: 全部样本训练数据，X.shape -> (n_x, m) n_x为特征维度，m为样本总数
        :param Y: 全部训练数据的标签 Y.shape -> (1, m)
        :return:
        """
        assert X.shape[1] == len(Y[0]), "the size of X_train muset be equal to y_train"

        m = X.shape[1]  # 训练样本数量

        # 前向传播. 1 -> hidden layer, 2 -> output layer
        Z1 = np.dot(w_1, X) + b_1  # X = [x1,x2,...,xm], Z1 = [z1,z2,...,zm]
        A1 = self._sigmoid(Z1)  # A1 = [a1, a2,...,am],选择隐藏层的激活函数

        Z2 = np.dot(w_2, A1) + b_2  # Z2 = [z1,z2,...,zm]
        A2 = self._sigmoid(Z2)  # A2 = [a1,a2,...,am] n_o=1的情况下，此时A2即为计算出来的预测结果

        # 反向传播，Back propagation 算法
        dz_2 = A2 - Y  # Y = [y1,...,ym] dz_2.shape -> (n_o, m)
        dw_2 = 1./m * np.dot(dz_2, A1.T)  # dw_2.shape -> (n_o, n_h)
        db_2 = 1./m * np.sum(dz_2, axis=1, keepdims=True)  # 输出为矩阵格式 db_2.shape -> (n_o, 1)

        dz_1 = np.dot(w_2.T, dz_2) * self._sigmoid_backford(Z1)  # dz_1.shape -> (n_h, m),隐藏层激活函数
        dw_1 = 1./m * np.dot(dz_1, X.T)  # dw_1.shape = (n_h, n_x)
        db_1 = 1./m * np.sum(dz_1, axis=1, keepdims=True)  # db_1.shape -> (n_h, 1)

        try:
            cost = -1./m * np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2))
            cost = np.squeeze(cost)
        except Exception as e:
            print(e)
            cost = float('inf')

        return dw_1, dw_2, db_1, db_2, cost

    def fit(self, X_train, y_train, n_iters=500, print_cost=True):
        """训练简单双层神经网络，采用sigmoid,relu,tanh激活函数，梯度下降法优化损失函数
        :param X_train: 训练数据集，X_train.shape -> (n_x, m)
        :param y_train: 训练数据集标签， y_train.shape -> (1, m)
        :param n_iters: 梯度下降法迭代优化次数
        :param print_cost: 是否输出每次迭代 cost func 的值
        :return: self
        """
        n_features, m = X_train.shape

        # 初始化权重系数和偏置值
        w_1 = np.random.random((self.hidden_layer_size[0], n_features)) * 0.01  # 初始化系数的选择 影响到激活函数收敛效果
        b_1 = np.zeros((self.hidden_layer_size[0], 1))
        w_2 = np.random.random((1, self.hidden_layer_size[0])) * 0.01
        b_2 = 0

        for i in range(n_iters):
            dw_1, dw_2, db_1, db_2, cost = self.propagate(X_train, y_train, w_1, b_1, w_2, b_2)
            w_1 -= self.learning_rate_init * dw_1
            b_1 -= self.learning_rate_init * db_1
            w_2 -= self.learning_rate_init * dw_2
            b_2 -= self.learning_rate_init * db_2

            if print_cost and i % 100 == 0:
                print("Cost after iteration {}:{}".format(i, cost))

        self.w_1 = w_1
        self.w_2 = w_2
        self.b_1 = b_1
        self.b_2 = b_2

        return self

    def predict_proba(self, X_predict):
        """输出预测概率矩阵"""
        assert self.w_1 is not None and self.w_2 is not None, "must fit before predict!"

        Z1 = self.w_1.dot(X_predict) + self.b_1  # X = [x1,x2,...,xm], Z1 = [z1,z2,...,zm]
        A1 = self._g(Z1, self.activation)

        Z2 = self.w_2.dot(A1) + self.b_2  # Z2 = [z1,z2,...,zm]
        A2 = self._g(Z2, self.activation)  # A2 = [a1,a2,...,am] n_o=1的情况下，此时A2即为计算出来的预测结果
        return A2

    def predict(self, X_predict):
        """输出预测结果矩阵"""
        assert self.w_1 is not None and self.w_2 is not None, "must fit before predict!"

        proba = self.predict_proba(X_predict)
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

    time0 = time.time()

    snn = SimpleNeuralNetwork(activation='sigmoid', hidden_layer_size=(4, 1), learning_rate_init=0.001)
    snn.fit(X_train, y_train)
    # print('预测结果概率矩阵:', snn.predict_proba(X_test))
    # print('预测结果：', snn.predict(X_test))
    print('模型准确率', snn.score(X_test, y_test))
    time1 = time.time()
    print('cost time:', time1-time0)

    print("\n-------sklearn中的效果------")
    X_train = X_train.T
    X_test = X_test.T
    y_train = y_train.ravel()
    y_test = y_test.ravel()
    mlp = MLPClassifier(solver='sgd', activation='logistic', alpha=1e-4, hidden_layer_sizes=(4,))
    mlp.fit(X_train, y_train)
    print("loss", mlp.loss_)
    print("模型准确率：", mlp.score(X_test, y_test))


