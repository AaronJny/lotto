# -*- coding: utf-8 -*-
# @File  : train.py
# @Author: AaronJny
# @Date  : 2019/10/29
# @Desc  :
import numpy as np
from models import model
from dataset import LottoDataSet
import settings
import utils


def simulate(test_np_x, test_np_y):
    """
    模拟购买彩票，对测试数据进行回测
    :param test_np_x: 测试数据输入
    :param test_np_y: 测试数据输出
    :return: 本次模拟的净收益
    """
    # 获得的奖金总额
    money_in = 0
    # 买彩票花出去的钱总额
    money_out = 0
    # 预测
    predicts = model.predict(test_np_x, batch_size=settings.BATCH_SIZE)
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
            # 存放每个球的概率分布的list
            probabilities = []
            # 对于每一种球，将其概率分布加入到列表中去
            for i in range(settings.FRONT_SIZE + settings.BACK_SIZE):
                probabilities.append(predicts[i][j])
            # 根据概率分布随机选择一个序列
            balls = utils.select_seqs(probabilities)
            # 计算奖金
            award = utils.lotto_calculate(outputs, balls)
            money_in += award
            if award:
                print('{} 中奖了,{}元！ {}/{}'.format(j, award, money_in, money_out))
    print('买彩票花费金钱共{}元，中奖金额共{}元，赚取{}元'.format(money_out, money_in, money_in - money_out))
    return money_in - money_out


# 初始化数据集
lotto_dataset = LottoDataSet(train_data_rate=0.9)
# 开始训练
results = []
for epoch in range(1, settings.EPOCHS + 1):
    model.fit(lotto_dataset.train_np_x, lotto_dataset.train_np_y, batch_size=settings.BATCH_SIZE, epochs=1)
    print('已训练完第{}轮，尝试模拟购买彩票...'.format(epoch))
    results.append(simulate(lotto_dataset.test_np_x, lotto_dataset.test_np_y))
