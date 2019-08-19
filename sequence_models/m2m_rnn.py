import numpy as np
import copy


class Many2ManyRNN:
    """ 实现二进制加法，通过上一位预测下一位的值
        Pred [0 1 0 0 0 0 1 1]
        True [0 1 0 0 0 0 1 1]
        44 + 23 = 67
        没有使用非线性激活函数
    """

    def __init__(self, alpha=0.05, input_dim=2, hidden_dim=16, output_dim=1, n_iters=100000):
        self._alpha = alpha
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._output_dim = output_dim
        self._n_iters = n_iters

    def _init_weights(self):
        """ 初始化权重系数
        :return:
        """
        w_1 = 2*np.random.random((self._input_dim, self._hidden_dim)) - 1
        w_h = 2*np.random.random((self._hidden_dim, self._hidden_dim)) - 1
        w_2 = 2 * np.random.random((self._hidden_dim, self._output_dim)) - 1

        w_1_update = np.zeros_like(w_1)
        w_h_update = np.zeros_like(w_h)
        w_2_update = np.zeros_like(w_2)

        return w_1, w_h, w_2, w_1_update, w_h_update, w_2_update

    @staticmethod
    def _sigmoid(z):
        return 1./(1+np.exp(-z))

    @staticmethod
    def _sigmoid_backward(A):
        return A*(1-A)

    def _generate_problem(self, ini2binary):
        """ 随机生成一个加法问题，并映射成二进制格式,(c = a + b)
        :param ini2binary: dict, 整数to二进制数查找表
        :return: a, b, c, res(存储最终预测结果)
        """
        binary_dim = 8
        largest_number = pow(2, binary_dim)
        # 生成两个二进制整数
        a_int = np.random.randint(largest_number / 2)  # int version
        a = ini2binary[a_int]  # binary encoding
        b_int = np.random.randint(largest_number / 2)  # int version
        b = ini2binary[b_int]  # binary encoding
        # true answer
        c_int = a_int + b_int
        c = ini2binary[c_int]
        # where we'll store our best guess (binary encoded)
        res = np.zeros_like(c)

        return a_int, a, b_int, b, c, res

    def fit(self, ini2binary):
        """ 输入数据（这里是二进制查找表),训练模型，得到预测表
        :param X: 整数to二进制数查找表
        :return:
        """
        w_1, w_h, w_2, w_1_update, w_h_update, w_2_update = self._init_weights()
        binary_dim = 8
        for i in range(self._n_iters):
            a_int, a_n, b_int, b_n, c_n, res = self._generate_problem(ini2binary)
            # 初始化损失值
            cost = 0

            layer_2_deltas = list()
            a = list()
            y_hat = list()
            a.append(np.zeros(self._hidden_dim))

            # 前向传播，从左到右
            for t in range(binary_dim):
                # generate input and output
                X = np.array([[a_n[binary_dim - t - 1], b_n[binary_dim - t - 1]]])
                y = np.array([[c_n[binary_dim - t - 1]]]).T

                # 一次前向传播，
                z_1 = np.dot(X, w_1) + np.dot(a[-1], w_h)
                a_1 = self._sigmoid(z_1)
                z_2 = np.dot(a_1, w_2)
                a_2 = self._sigmoid(z_2)
                # 一次反向传播
                layer_2_error = y - a_2
                layer_2_deltas.append((layer_2_error)*self._sigmoid_backward(a_2))

                cost += - np.sum(y * np.log(a_2) + (1 - y) * np.log(1 - a_2))
                # cost += np.abs(layer_2_error[0])

                # 预测值
                res[binary_dim - t -1] = np.round(a_2[0][0])
                # 存储隐藏层，每个时间t的a
                a.append(copy.deepcopy(a_1))

            future_layer_h_delta = np.zeros(self._hidden_dim)

            # 反向传播，从右向左
            for t in range(binary_dim):
                X = np.array([[a_n[t], b_n[t]]])
                layer_h = a[-t - 1]
                prev_layer_h = a[-t - 2]
                # 反向传播
                layer_2_delta = layer_2_deltas[-t - 1]
                layer_h_delta = (future_layer_h_delta.dot(w_h.T) + layer_2_delta.dot(
                    w_2.T)) * self._sigmoid_backward(layer_h)
                # 计算参数导数
                w_2_update += np.atleast_2d(layer_h).T.dot(layer_2_delta)
                w_h_update += np.atleast_2d(prev_layer_h).T.dot(layer_h_delta)
                w_1_update += X.T.dot(layer_h_delta)

                future_layer_h_delta = layer_h_delta
            # 梯度下降法更新参数
            w_1 += self._alpha * w_1_update
            w_h += self._alpha * w_h_update
            w_2 += self._alpha * w_2_update

            w_1_update *= 0
            w_h_update *= 0
            w_2_update *= 0

            # 输出训练结果
            if i % 1000 == 0:
                print("cost {} after {} iters.".format(cost, i))
                print("Pred", res)
                print("True", c_n)
                out = 0
                for index, x in enumerate(reversed(res)):
                    out += x * pow(2, index)
                print(str(a_int) + " + " + str(b_int) + " = " + str(out))
                print("------------")


if __name__ == '__main__':

    def training_dataset_generation():
        # training dataset generation, 构建整数到二进制数的查找表
        int2binary = {}
        # 八位二进制数
        binary_dim = 8
        largest_number = pow(2, binary_dim)
        # 构建整数到二进制数的映射查找表
        binary = np.unpackbits(
            np.array([range(largest_number)], dtype=np.uint8).T, axis=1)
        for i in range(largest_number):
            int2binary[i] = binary[i]

        return int2binary

    ini2binary = training_dataset_generation()
    brnn = Many2ManyRNN()
    brnn.fit(ini2binary)


