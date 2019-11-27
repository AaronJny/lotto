# -*- coding: utf-8 -*-
# @File  : random_show.py
# @Author: AaronJny
# @Date  : 2019/10/29
# @Desc  : 随机选择情况下的收益情况
import random
import numpy as np
from dataset import LottoDataSet
import utils
import settings


def get_one_random_sample():
    """
    获取一种随机序列
    :return:
    """
    front_balls = list(range(settings.FRONT_VOCAB_SIZE))
    back_balls = list(range(settings.BACK_VOCAB_SIZE))
    return random.sample(front_balls, settings.FRONT_SIZE) + random.sample(back_balls, settings.BACK_SIZE)


def simulate(test_np_x, test_np_y):
    # 获得的奖金总额
    money_in = 0
    # 买彩票花出去的钱总额
    money_out = 0
    # 共有多少组数据
    samples_num = len(test_np_x['x1'])
    # 对于每一组数据
    for j in range(samples_num):
        # 这一期的真实开奖结果
        outputs = []
        for k in range(settings.FRONT_SIZE + settings.BACK_SIZE):
            outputs.append(np.argmax(test_np_y['y{}'.format(k + 1)][j]))
        # 每一期彩票买五注
        money_out += 10
        for k in range(5):
            balls = get_one_random_sample()
            # 计算奖金
            award = utils.lotto_calculate(outputs, balls)
            money_in += award
    print('买彩票花费金钱共{}元，中奖金额共{}元，赚取{}元'.format(money_out, money_in, money_in - money_out))
    return money_in - money_out


dataset = LottoDataSet(train_data_rate=0.9)
# 随机买一百次，并记录每一次收入-支出的差值
results = []
for epoch in range(1, 101):
    results.append(simulate(dataset.test_np_x, dataset.test_np_y))
# 去除最高的和最低的
results = sorted(results)[1:-1]
# 计算平均值
print('mean', sum(results) / len(results))
