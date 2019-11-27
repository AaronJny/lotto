# -*- coding: utf-8 -*-
# @File  : utils.py
# @Author: AaronJny
# @Date  : 2019/10/29
# @Desc  : 对数据进行处理和操作的一些工具方法
import matplotlib.pyplot as plt
import numpy as np
import settings


def sample(preds, temperature=1.0):
    """
    从给定的preds中随机选择一个下标。
    当temperature固定时，preds中的值越大，选择其下标的概率就越大；
    当temperature不固定时，
        temperature越大，选择值小的下标的概率相对提高，
        temperature越小，选择值大的下标的概率相对提高。
    :param preds: 概率分布序列，其和为1.0
    :param temperature: 当temperature==1.0时，相当于直接对preds进行轮盘赌法
    :return:
    """
    preds = np.asarray(preds).astype(np.float64)
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def search_award(front_match_num, back_match_num, cache={}):
    """
    给定前后区命中数量，使用记忆化搜索查找并计算对应奖金
    :param front_match_num: 前区命中数量
    :param back_match_num: 后区命中数量
    :param cache: 缓存用的记忆字典
    :return:
    """
    # 前后区都没有命中，奖金为0
    if front_match_num == 0 and back_match_num == 0:
        return 0
    # 尝试直接从缓存里面获取奖金
    award = cache.get((front_match_num, back_match_num), -1)
    # 这里使用-1是为了避免0和None在判断上的混淆
    # 如果缓存里面有，已经计算过了，就直接返回
    if award != -1:
        return award
    # 尝试直接从中奖规则中获取奖金数量
    award = settings.AWARD_RULES.get((front_match_num, back_match_num), -1)
    if award == -1:
        # 如果没找到，就先认为没中奖，然后将前区命中数量或后区命中数量减一，
        # 递归查找，保留最大的中奖金额
        award = 0
        if front_match_num > 0:
            award = search_award(front_match_num - 1, back_match_num)
        if back_match_num > 0:
            award = max(award, search_award(front_match_num, back_match_num - 1))
    # 缓存下本次计算结果
    cache[(front_match_num, back_match_num)] = award
    # 返回奖金数额
    return award


def lotto_calculate(winning_sequence, sequence_selected):
    """
    给定中奖序列和选择的序列，计算获奖金额
    :param winning_sequence:中奖序列
    :param sequence_selected: 选择的序列
    :return:
    """
    # 前区命中数量
    front_match = len(
        set(winning_sequence[:settings.FRONT_SIZE]).intersection(set(sequence_selected[:settings.FRONT_SIZE])))
    # 后区命中数量
    back_match = len(
        set(winning_sequence[settings.FRONT_SIZE:]).intersection(set(sequence_selected[settings.FRONT_SIZE:])))
    # 计算奖金
    award = search_award(front_match, back_match)
    return award


def select_seqs(predicts):
    """
    根据给定的概率分布，随机选择一种彩票序列
    :param predicts:list[list] 每一个球的概率分布组成的列表
    :return: list 彩票序列
    """
    balls = []
    # 对于每一种球
    for predict in predicts:
        try_cnt = 0
        while True:
            try_cnt += 1
            # 根据预测结果随机选择一个
            if try_cnt < 100:
                ball = sample(predict)
            else:
                # 如果连续100次都是重复的，就等概率地从所有球里面选择一个
                ball = sample([1. / len(predict) for __ in predict])
            # 如果选重复了就重新选
            if ball in balls:
                # 序列不长，就没有使用set优化，直接用list了
                continue
            # 将球保存下来，跳出，开始选取下一个
            balls.append(ball)
            break
    # 排序，前五个升序，后两个升序
    balls = sorted(balls[:settings.FRONT_SIZE]) + sorted(balls[settings.FRONT_SIZE:])
    return balls


def draw_graph(y):
    """
    绘制给定列表y的折线图和趋势线
    """
    # 横坐标，第几轮训练
    x = list(range(len(y)))
    # 拟合一次函数，返回函数参数
    parameter = np.polyfit(x, y, 1)
    # 拼接方程
    f = np.poly1d(parameter)
    # 绘制图像
    plt.plot(x, f(x), "r--")
    plt.plot(y)
    plt.show()
