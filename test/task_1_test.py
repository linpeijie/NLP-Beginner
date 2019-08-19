import unittest
from task.task_1 import Task1


class Task1test(unittest.TestCase):

    def test_data(self):
        task = Task1()

        X, y = task.load_dataset()
        self.assertEqual(X[0], 34345, msg='数据错误')

        train = task.BoW(X)
        self.assertEqual(train[1], 11757, msg='数据错误')
 
    def test_sag_C(self):
        """
        未做特征选择，选用sag来优化函数，会遇到无法拟合的问题
        实验目的：分析在相同特征，相同损失函数，在不同学习率C下的最终分类性能
        前置条件：penalty 为L2范数，算法收敛最大迭代次数100
        :return:
        """
        task = Task1()

        X, y = task.load_dataset()
        train = task.BoW(X)

        train, y = train[:200], y[:200]

        print("------C为1.0-------")
        task.lr(train, y, seed=666, solver="sag", C=1.0, max_iter=3000)

        print("------C为0.8-------")
        task.lr(train, y, seed=666, solver="sag", C=0.5, max_iter=3000)

        print("------C为0.5-------")
        task.lr(train, y, seed=666, solver="sag", C=0.1, max_iter=3000)
