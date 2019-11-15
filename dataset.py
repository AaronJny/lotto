# -*- coding: utf-8 -*-
# @File  : dataset.py
# @Author: AaronJny
# @Date  : 2019/10/29
# @Desc  : 对数据集进行相关处理
import time
import numpy as np
import settings


class LottoDataSet:

    def __init__(self, path=settings.DATASET_PATH, train_data_rate=0.9):
        # 数据集路径
        self.path = path
        # 训练集占整体数据集的比例
        self.train_data_rate = train_data_rate
        # 训练集
        self.train_np_x = {}
        self.train_np_y = {}
        # 测试集
        self.test_np_x = {}
        self.test_np_y = {}
        # 加载并处理数据集
        self.clean_data()

    def load_data_from_path(self, path=None):
        """
        从给定路径加载数据集
        :param path: 数据集路径
        :return: list,若干组开奖记录
        """
        # 如果没有指定路径，就是用初始化实例时传递的path
        if not path:
            path = self.path
        # 读取数据行
        with open(path) as f:
            # 因为csv里面最新的数据放在了最前面，所以我们需要颠倒一下
            lines = f.readlines()[::-1]
            # 排除空行
            lines = [line.strip() for line in lines if line.strip()]
        return lines

    def clean_data(self):
        """
        对数据进行清洗
        :return:
        """
        # 先从硬盘读取文件
        lines = self.load_data_from_path()
        # 去除引号，并使用逗号分割，将数据转成数组
        x_nums = []
        for line in lines:
            # 下标0的位置是期号，直接去掉
            nums = line.replace('"', '').split(',')[1:]
            # 所有球的编号都减一，把1-35变成0-34,1-12变成0-11
            # 这样便于后面做softmax
            x_nums.append([int(x) - 1 for x in nums])
        # 接着，把中奖序列中的七个数字拆开，按位置和时间纵轴组合，变成7组数据
        num_seqs = {}
        # 对于每一期的中奖序列
        for line in x_nums:
            # 对于一条中奖序列中的每一个数
            for index, num in enumerate(line):
                # 最后的数据格式{0: [1,2,3,4,...],1: [1,2,3,4,...],...,6: [1,2,3,4,...]}
                num_seqs.setdefault(index, []).append(num)
        # 根据时间序列，拆出来x和y数据集，每MAX_STEPS长度的连续序列构成一条数据的x，max_steps+1构成y
        # 举例，假设MAX_STEPS=3，有序列[1,2,3,4,5,6],则[1,2,3->4],[2,3,4->5],[3,4,5->6]是组成的数据集
        x = {}
        y = {}
        for index, seqs in num_seqs.items():
            x[index] = []
            y[index] = []
            total = len(seqs)
            # 序列的长度要求为MAX_STEPS,所以从MAX_STEPS处开始，而不是下标0处
            for i in range(settings.MAX_STEPS, total, 1):
                # 存放本条x序列的，存放的是数字形式
                tmp_x = []
                # 存放本条y值的，存放的one-hot形式，虽然y只是一个数，但one-hot形式也为list
                if index < settings.FRONT_SIZE:
                    # 根据index判断当前号码属于前区还是后区，使用相关的号码种类数量来初始化one-hot向量
                    # 因为前区是35选5，后区是12选2，one-hot向量大小不同，所以要区别对待
                    tmp_y = [0 for _ in range(settings.FRONT_VOCAB_SIZE)]
                else:
                    tmp_y = [0 for _ in range(settings.BACK_VOCAB_SIZE)]
                # 将从i-MAX_STEPS到i(不包括i)的这一段长为MAX_STEPS的序列，逐个加入到tmp_x中
                for j in range(i - settings.MAX_STEPS, i, 1):
                    tmp_x.append(seqs[j])
                # 将这条记录添加到x数据集中
                x[index].append(tmp_x)
                # 修改y值的one-hot，并将标签加入到y数据集中
                tmp_y[seqs[i]] = 1
                y[index].append(tmp_y)
        # y在前面已经是one-hot形式了，我们现在需要把x里面的数字也转成one-hot形式，并转成numpy的array类型
        np_x = {}
        np_y = {}
        # 对应7个球构成的七组序列中的每一组
        for i in range(settings.FRONT_SIZE + settings.BACK_SIZE):
            # 获取样本数量
            x_len = len(x[i])
            # 根据球所处的前后区，分别进行初始化
            if i < settings.FRONT_SIZE:
                tmp_x = np.zeros((x_len, settings.MAX_STEPS, settings.FRONT_VOCAB_SIZE))
                tmp_y = np.zeros((x_len, settings.FRONT_VOCAB_SIZE))
            else:
                tmp_x = np.zeros((x_len, settings.MAX_STEPS, settings.BACK_VOCAB_SIZE))
                tmp_y = np.zeros((x_len, settings.BACK_VOCAB_SIZE))
            # 分别利用x,y中的数据修改tmp_x和tmp_y
            for j in range(x_len):
                for k, num in enumerate(x[i][j]):
                    tmp_x[j][k][num] = 1
                for k, num in enumerate(y[i][j]):
                    tmp_y[j][k] = num
            # 然后将tmp_x和tmp_y按照球所处的位置，加入到np_x和np_y中
            np_x['x{}'.format(i + 1)] = tmp_x
            np_y['y{}'.format(i + 1)] = tmp_y
        # ok,现在我们可以看一下数组的shape是否正确
        for i in range(settings.FRONT_SIZE + settings.BACK_SIZE):
            print(i + 1, np_x['x{}'.format(i + 1)].shape, np_y['y{}'.format(i + 1)].shape)
        # 划分数据集
        total_batch = len(np_x['x1'])  # 总样本数
        train_batch_num = int(total_batch * self.train_data_rate)  # 训练样本数
        train_np_x = {}
        train_np_y = {}
        test_np_x = {}
        test_np_y = {}
        for i in range(settings.FRONT_SIZE + settings.BACK_SIZE):
            x_index = 'x{}'.format(i + 1)
            y_index = 'y{}'.format(i + 1)
            train_np_x[x_index] = np_x[x_index][:train_batch_num]
            train_np_y[y_index] = np_y[y_index][:train_batch_num]
            test_np_x[x_index] = np_x[x_index][train_batch_num:]
            test_np_y[y_index] = np_y[y_index][train_batch_num:]
        # 打乱训练数据
        random_seed = int(time.time())
        # 使用相同的随机数种子，保证x和y的一一对应没有被破坏
        for i in range(settings.FRONT_SIZE + settings.BACK_SIZE):
            np.random.seed(random_seed)
            np.random.shuffle(train_np_x['x{}'.format(i + 1)])
            np.random.seed(random_seed)
            np.random.shuffle(train_np_y['y{}'.format(i + 1)])
        self.train_np_x = train_np_x
        self.train_np_y = train_np_y
        self.test_np_x = test_np_x
        self.test_np_y = test_np_y
