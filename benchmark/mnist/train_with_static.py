#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
  * @file train_with_static.py
  * @author zhengtianyu
  * @date 2019/7/26 2:10 PM
  * @brief 
  *
  **************************************************************************/
"""
from __future__ import print_function
import argparse
import ast
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.optimizer import AdamOptimizer
from paddle.fluid.dygraph.base import to_variable
from model import MNIST


def parse_args():
    parser = argparse.ArgumentParser("Training for Mnist.")
    parser.add_argument(
        "--use_data_parallel",
        type=ast.literal_eval,
        default=False,
        help="The flag indicating whether to shuffle instances in each pass.")
    parser.add_argument("-e", "--epoch", default=5, type=int, help="set epoch")
    parser.add_argument("--ce", action="store_true", help="run ce")
    args = parser.parse_args()
    return args


def train_mnist(args):
    epoch_num = args.epoch
    BATCH_SIZE = 64
    seed = 33
    np.random.seed(seed)
    start_prog = fluid.Program()
    main_prog = fluid.Program()
    start_prog.random_seed = seed
    main_prog.random_seed = seed
    with fluid.program_guard(main_prog, start_prog):
        exe = fluid.Executor(fluid.CPUPlace())
        mnist = MNIST("mnist")
        adam = AdamOptimizer(learning_rate=0.001)
        train_reader = paddle.batch(
            paddle.dataset.mnist.train(), batch_size=BATCH_SIZE, drop_last=True)
        img = fluid.layers.data(
            name='pixel', shape=[1, 28, 28], dtype='float32')
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        cost = mnist(img)
        loss = fluid.layers.cross_entropy(cost, label)
        avg_loss = fluid.layers.mean(loss)
        adam.minimize(avg_loss)
        out = exe.run(fluid.default_startup_program())
        for epoch in range(epoch_num):
            for batch_id, data in enumerate(train_reader()):
                static_x_data = np.array(
                    [x[0].reshape(1, 28, 28)
                     for x in data]).astype('float32')
                y_data = np.array(
                    [x[1] for x in data]).astype('int64').reshape([BATCH_SIZE, 1])

                fetch_list = [avg_loss.name]
                out = exe.run(
                    fluid.default_main_program(),
                    feed={"pixel": static_x_data,
                          "label": y_data},
                    fetch_list=fetch_list)

                static_out = out[0]

                if batch_id % 100 == 0:
                    print("epoch: {}, batch_id: {}, loss: {}".format(epoch, batch_id, static_out))



if __name__ == '__main__':
    args = parse_args()
    train_mnist(args)