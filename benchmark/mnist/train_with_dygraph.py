#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
  * @file train_with_dygraph.py
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
    place = fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        seed = 33
        np.random.seed(seed)
        fluid.default_startup_program().random_seed = seed
        fluid.default_main_program().random_seed = seed

        if args.use_data_parallel:
            strategy = fluid.dygraph.parallel.prepare_context()
        mnist = MNIST("mnist")
        adam = AdamOptimizer(learning_rate=0.001)
        if args.use_data_parallel:
            mnist = fluid.dygraph.parallel.DataParallel(mnist, strategy)

        train_reader = paddle.batch(
            paddle.dataset.mnist.train(), batch_size=BATCH_SIZE, drop_last=True)
        if args.use_data_parallel:
            train_reader = fluid.contrib.reader.distributed_batch_reader(
                train_reader)

        test_reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=BATCH_SIZE, drop_last=True)

        for epoch in range(epoch_num):
            for batch_id, data in enumerate(train_reader()):
                dy_x_data = np.array([x[0].reshape(1, 28, 28)
                                      for x in data]).astype('float32')
                y_data = np.array(
                    [x[1] for x in data]).astype('int64').reshape(-1, 1)

                img = to_variable(dy_x_data)
                label = to_variable(y_data)
                label.stop_gradient = True

                cost, acc = mnist(img, label)

                loss = fluid.layers.cross_entropy(cost, label)
                avg_loss = fluid.layers.mean(loss)

                if args.use_data_parallel:
                    avg_loss = mnist.scale_loss(avg_loss)
                    avg_loss.backward()
                    mnist.apply_collective_grads()
                else:
                    avg_loss.backward()

                adam.minimize(avg_loss)
                # save checkpoint
                mnist.clear_gradients()
                if batch_id % 100 == 0:
                    print("Loss at epoch {} step {}: {:}".format(
                        epoch, batch_id, avg_loss.numpy()))
        print("checkpoint saved")



if __name__ == '__main__':
    args = parse_args()
    train_mnist(args)