# -*- coding: utf-8 -*-
# @File  : models.py
# @Author: AaronJny
# @Date  : 2019/10/29
# @Desc  : 建立深度学习模型
import keras
from keras import layers
from keras import models
import settings

# 这是一个多输入模型，inputs用来保存所有的输入层
inputs = []
# 这是一个多输出模型，outputs用来保存所有的输出层
outputs = []
# 前区的中间层列表，用于拼接
front_temps = []
# 后区的中间层
back_temps = []

# 处理前区的输入变换
for i in range(settings.FRONT_SIZE):
    # 输入层
    x_input = layers.Input((settings.MAX_STEPS, settings.FRONT_VOCAB_SIZE), name='x{}'.format(i + 1))
    # 双向循环神经网络
    x = layers.Bidirectional(layers.LSTM(settings.LSTM_UNITS, return_sequences=True))(x_input)
    # 随机失活
    x = layers.Dropout(rate=settings.DROPOUT_RATE)(x)
    x = layers.Bidirectional(layers.LSTM(settings.LSTM_UNITS, return_sequences=True))(x)
    x = layers.Dropout(rate=settings.DROPOUT_RATE)(x)
    x = layers.TimeDistributed(layers.Dense(settings.FRONT_VOCAB_SIZE * 3))(x)
    # 平铺
    x = layers.Flatten()(x)
    # 全连接
    x = layers.Dense(settings.FRONT_VOCAB_SIZE * 3, activation='relu')(x)
    # 保存输入层
    inputs.append(x_input)
    # 保存前区中间层
    front_temps.append(x)
# 处理后区的输入和变换
for i in range(settings.BACK_SIZE):
    # 输入层
    x_input = layers.Input((settings.MAX_STEPS, settings.BACK_VOCAB_SIZE),
                           name='x{}'.format(i + 1 + settings.FRONT_SIZE))
    # 双向循环神经网络
    x = layers.Bidirectional(layers.LSTM(settings.LSTM_UNITS, return_sequences=True))(x_input)
    # 随机失活
    x = layers.Dropout(rate=settings.DROPOUT_RATE)(x)
    x = layers.Bidirectional(layers.LSTM(settings.LSTM_UNITS, return_sequences=True))(x)
    x = layers.Dropout(rate=settings.DROPOUT_RATE)(x)
    x = layers.TimeDistributed(layers.Dense(settings.BACK_VOCAB_SIZE * 3))(x)
    # 平铺
    x = layers.Flatten()(x)
    # 全连接
    x = layers.Dense(settings.BACK_VOCAB_SIZE * 3, activation='relu')(x)
    # 保存输入层
    inputs.append(x_input)
    # 保存后区中间层
    back_temps.append(x)
# 连接
front_concat_layer = layers.concatenate(front_temps)
back_concat_layer = layers.concatenate(back_temps)
# 使用softmax计算分布概率
for i in range(settings.FRONT_SIZE):
    x = layers.Dense(settings.FRONT_VOCAB_SIZE, activation='softmax', name='y{}'.format(i + 1))(front_concat_layer)
    outputs.append(x)
for i in range(settings.BACK_SIZE):
    x = layers.Dense(settings.BACK_VOCAB_SIZE, activation='softmax', name='y{}'.format(i + 1 + settings.FRONT_SIZE))(
        back_concat_layer)
    outputs.append(x)
# 创建模型
model = models.Model(inputs, outputs)
# 指定优化器和损失函数
model.compile(optimizer=keras.optimizers.Adam(),
              loss=[keras.losses.categorical_crossentropy for __ in range(settings.FRONT_SIZE + settings.BACK_SIZE)],
              loss_weights=[0.1, 0.1, 0.1, 0.1, 0.1, 0.25, 0.25])
# 查看网络结构
model.summary()

# 可视化模型，保存结构图
# from keras.utils import plot_model
# plot_model(model, to_file='model.png')
