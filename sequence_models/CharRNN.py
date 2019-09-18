"""
    nltk的names数据集，CharRNN简单的做文本生成任务
"""
# coding=utf-8
from torch import nn
import torch as t
import torch.nn.functional as F
import numpy as np
import sklearn
import string
import random
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('names')
from nltk.corpus import names


use_cuda = t.cuda.is_available()
device = t.device("cuda" if use_cuda else "cpu")

chars = string.ascii_lowercase + '-' + ' ' + "'" +'!'

hiddem_dim = 256
vocab_size = len(chars)


def name2input(name):
    """ one-hot向量，名字向量化
    :param name:
    :return:
    """
    ids = [chars.index(c) for c in name if c not in["\\"]]
    a = np.zeros(shape=(len(ids), len(chars)), dtype=np.long)
    for i, idx in enumerate(ids):
        a[i][idx] = 1

    return a


def name2target(name):
    ids = [chars.index(c) for c in name if c not in ["\\"]]
    return ids


def sexy2input(sexy):
    a = np.zeros(shape=(1, 2), dtype=np.long)
    a[0][sexy] = 1
    return a


def load_data():
    """ 加载数据
    :return: list[(name, 0), (name, 1)...]
    """
    female_file, male_file = names.fileids()

    female = names.words(female_file)
    male = names.words(male_file)

    data_set = [(name.lower(), 0) for name in female] + [(name.lower(), 1) for name in male]
    random.shuffle(data_set)
    print('10 names:', data_set[:10])
    return data_set


class CharRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        """
        :param vocab_size: 全部字符的数量
        :param hidden_dim: 隐藏元的神经元个数
        :param embedding_dim: 每个字符的维度
        """
        super(CharRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.output_size = vocab_size
        # 输入维度增加了性别，故+2
        self.rnn = nn.GRU(embedding_dim+2, hidden_dim, batch_first=True)
        self.liner = nn.Linear(hidden_dim, vocab_size)

    def forward(self, sexy, name, hidden=None):
        if hidden is None:
            h0 = t.randn(1, 1, self.hidden_dim, device=device) * 0.01
        else:
            h0 = hidden

        input = t.cat([sexy, name], dim=2)

        output, hidden = self.rnn(input, h0)
        output = self.liner(output)
        output = F.dropout(output, 0.3)
        output = F.softmax(output, dim=2)

        return output.view(1, -1), hidden


class Model:
    def __init__(self, epoches=10):
        self.model = CharRNN(vocab_size, len(chars), hiddem_dim)
        self.model.to(device)
        self.epoches = epoches

    def train(self, train_set):
        optimizer = t.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.epoches):
            total_loss = 0
            for x in range(1000):
                loss = 0  # 初始化

                name, sexy = random.choice(train_set)
                optimizer.zero_grad()
                hidden = t.zeros(1, 1, hiddem_dim, device=device)

                for x, y in zip(list(name), list(name[1:] + '!')):
                    name_tensor = t.tensor([name2input(x)], dtype=t.float, device=device)
                    sexy_tensor = t.tensor([sexy2input(sexy)], dtype=t.float, device=device)
                    target_tensor = t.tensor(name2target(y), dtype=t.long, device=device)

                    output, _ = self.model(sexy_tensor, name_tensor, hidden)
                    loss += criterion(output, target_tensor)

                loss.backward()
                optimizer.step()

                total_loss += loss/(len(name) - 1)

            print("Training: in epoch {} loss {}".format(epoch, total_loss/1000))

    def sample(self, sexy, start):
        max_len = 8
        result = []

        with t.no_grad():
            hidden = None
            for c in start:
                name_tensor = t.tensor([name2input(c)], dtype=t.float, device=device)
                sexy_tensor = t.tensor([sexy2input(sexy)], dtype=t.float, device=device)
                output, _ = self.model(sexy_tensor, name_tensor, hidden)

            c = start[-1]

            while c != '!':
                name_tensor = t.tensor([name2input(c)], dtype=t.float, device=device)
                sexy_tensor = t.tensor([sexy2input(sexy)], dtype=t.float, device=device)
                output, _ = self.model(sexy_tensor, name_tensor, hidden)
                topv, topi = output.topk(1)
                c = chars[topi]

                result.append(c)

                if len(result) > max_len:
                    break

        return start + "".join(result[:-1])


if __name__ == "__main__":
    model = Model()
    data_set = load_data()

    model.train(data_set)
    print(model.sample(0, 'j'))
    c = input('please input name prefix:')
    while c != 'q':
        print(model.sample(1, c))
        print(model.sample(0, c))
        c = input('please input name prefix:')
