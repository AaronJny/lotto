# -*- coding: utf-8 -*-
# @File    : predict.py
# @Author  : AaronJny
# @Date    : 2019/11/26
# @Desc    : 指定一个训练好的模型参数，让模型随机选出下期彩票号码
from dataset import LottoDataSet
from models import model
import settings
import utils

# 加载模型参数
model.load_weights(settings.PREDICT_MODEL_PATH)
# 构建数据集
lotto_dataset = LottoDataSet()
# 提取倒数第MAX_STEPS期到最近一期的数据，作为预测的输入
x = lotto_dataset.predict_data
# 开始预测
predicts = model.predict(x, batch_size=1)
# 存放选号结果的列表
result = []
# 存放每个球的概率分布的list
probabilities = [predict[0] for predict in predicts]
# print(probabilities)
# 总共要选出settings.PREDICT_NUM注彩票
for i in range(settings.PREDICT_NUM):
    # 根据概率分布随机选择一个序列
    balls = utils.select_seqs(probabilities)
    # 加入到选号列表中,注意，我们需要把全部的数字+1，恢复原始的编号
    result.append([ball + 1 for ball in balls])
# 输出要买的彩票序列
print('本次预测结果如下：')
for index, balls in enumerate(result, start=1):
    print('第{}注 {}'.format(index, ' '.join(map(str, balls))))
