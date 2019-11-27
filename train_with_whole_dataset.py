# -*- coding: utf-8 -*-
# @File    : train_with_whole_dataset.py
# @Author  : AaronJny
# @Date    : 2019/11/26
# @Desc    : 使用全部数据集进行训练
import os
from models import model
from dataset import LottoDataSet
import settings

# 初始化数据集
lotto_dataset = LottoDataSet(train_data_rate=1)
# 创建保存权重的文件夹
if not os.path.exists(settings.CHECKPOINTS_PATH):
    os.mkdir(settings.CHECKPOINTS_PATH)
# 开始训练
model.fit(lotto_dataset.train_np_x, lotto_dataset.train_np_y, batch_size=settings.BATCH_SIZE, epochs=settings.EPOCHS)
# 保存模型
model.save_weights('{}/model_checkpoint_x'.format(settings.CHECKPOINTS_PATH))
