# -*- coding: utf-8 -*-
# @File  : settings.py
# @Author: AaronJny
# @Date  : 2019/10/29
# @Desc  :

# 训练epochs数量
EPOCHS = 60
# 训练批大小
BATCH_SIZE = 128
# 输入的连续时间序列长度
MAX_STEPS = 256
# 前区号码种类数
FRONT_VOCAB_SIZE = 35
# 后区号码种类数
BACK_VOCAB_SIZE = 12
# dropout随机失活比例
DROPOUT_RATE = 0.5
# 长短期记忆网络单元数
LSTM_UNITS = 64
# 前区需要选择的号码数量
FRONT_SIZE = 5
# 后区需要选择的号码数量
BACK_SIZE = 2
# 保存训练好的参数的路径
CHECKPOINTS_PATH = 'checkpoints'
# 预测下期号码时使用的训练好的模型参数的路径，默认使用完整数据集训练出的模型
PREDICT_MODEL_PATH = '{}/model_checkpoint_x'.format(CHECKPOINTS_PATH)
# 预测的时候，预测几注彩票,默认5注
PREDICT_NUM = 5
# 数据集路径
DATASET_PATH = 'lotto.csv'
# 数据集下载地址
LOTTO_DOWNLOAD_URL = 'https://www.js-lottery.com/PlayZone/downLottoData.html'
# 大乐透中奖及奖金规则（没有考虑浮动情况，税前）
AWARD_RULES = {
    (5, 2): 10000000,
    (5, 1): 800691,
    (5, 0): 10000,
    (4, 2): 3000,
    (4, 1): 300,
    (3, 2): 200,
    (4, 0): 100,
    (3, 1): 15,
    (2, 2): 15,
    (3, 0): 5,
    (2, 1): 5,
    (1, 2): 5,
    (0, 2): 5
}
