"""
    实现基于logistic/softmax regression的文本分类

    数据集：Classify the sentiment of sentences from the Rotten Tomatoes dataset

    实现要求：NumPy

    需要了解的知识点：
        文本特征表示：Bag-of-Word，N-gram
        分类器：logistic/softmax regression，损失函数、（随机）梯度下降、特征选择
        数据集：训练集/验证集/测试集的划分

    实验：
        分析不同的特征、损失函数、学习率对最终分类性能的影响
        shuffle 、batch、mini-batch
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression


class Task1:

    def __init__(self):
        pass

    def load_dataset(self):
        """
        获取原始数据，提取Phrase列和Sentiment列
        :return: X训练数据，y训练数据的标签，
        """
        names = ['PhraseId', 'SentenceId', 'Phrase', 'Sentiment']

        train = pd.read_csv('../datasets/sentiment-analysis-on-movie-reviews/train.tsv', sep='\t')
        test = pd.read_csv('../datasets/sentiment-analysis-on-movie-reviews/test.tsv', sep='\t')
        test_target = pd.read_csv('../datasets/sentiment-analysis-on-movie-reviews/sampleSubmission.csv')

        X = np.concatenate((np.array(train['Phrase']), np.array(test['Phrase'])), axis=0)
        y = np.concatenate((np.array(train['Sentiment']), np.array(test_target['Sentiment'])), axis=0)

        X = X[y < 2]
        y = y[y < 2]

        return X, y

    def BoW(self, text: list):
        """建立全部文件的词袋模型
        :param text:
        :return:
        """

        def stop_words():
            stop_word = []
            with open('../stop_words.txt', encoding='UTF-8') as fr:
                lines = fr.readlines()
                for line in lines:
                    stop_word.append(line.strip())
            return stop_word

        vectorizer = CountVectorizer(encoding='utf-8', stop_words=stop_words())
        vectorizer.fit(text)
        vector = vectorizer.transform(text)

        print(vector.shape)

        return vector.toarray()

    def lr(self, train, target, seed, solver, C, max_iter):

        X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.2, random_state=seed)

        # 对数据的训练集进行标准化
        ss = StandardScaler()
        X_train = ss.fit_transform(X_train)  # 先拟合数据在进行标准化

        # 训练模型
        lr = LogisticRegression(multi_class="ovr", penalty="l2", solver=solver, tol=1e-4, C=C,
                                max_iter=max_iter, n_jobs=4)
        re = lr.fit(X_train, y_train)

        # 模型效果获取
        r = re.score(X_train, y_train)
        print("R值(准确率):", r)
        print("稀疏化特征比率:%.2f%%" % (np.mean(lr.coef_.ravel() == 0) * 100))

        # 预测
        X_test = ss.transform(X_test)  # 数据标准化
        Y_predict = lr.predict(X_test)  # 预测

        print("=============准确率==============", '\n', metrics.accuracy_score(y_test.ravel(), Y_predict))
        print("=============召回率==============", '\n', metrics.recall_score(y_test.ravel(), Y_predict))
        print('training done!')


if __name__ == '__main__':
    task = Task1()
    X, y = task.load_dataset()
    task.BoW(X)
