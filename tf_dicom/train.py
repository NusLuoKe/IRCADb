#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/12 19:51
# @File    : train.py
# @Author  : NUS_LuoKe

from tf_dicom.load_dicom import *
import tensorflow as tf
from tf_dicom import model

base_dir = "F:/IRCAD/3Dircadb1/"

slice_path_list, liver_path_list = get_slice_liver_path(base_dir, shuffle=True)
training_set, validation_set, test_set = get_tra_val_test_set(slice_path_list, liver_path_list)

batch_size = 4
length = 224
width = 224
channel = 1


def train():
    # GPU limit
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.9

    # define placeholder
    x_img = tf.placeholder(tf.float32, shape=[batch_size, length, width, channel])
    y_true = tf.placeholder(tf.float32, shape=[batch_size, length, width, channel])

    # 1. Forward propagation
    pred = model.DenseNet(x_img, reduction=0.5)  # DenseNet_121(x_img, n_classes=3, is_train=True)
    y_pred = pred.outputs  # (512, 512, 1)

    # 2. loss
    loss = model.cross_entropy(y_true, y_pred)

    # 3. optimizer
    learning_rate = 1e-6
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    with tf.Session(config=config) as sess:
        # initial  variables
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # set epochs
        nb_epoch = 20
        for epoch in range(nb_epoch):
            print("EPOCH=%s:" % epoch)
            step = 0
            for batch_x_y in get_batch(slice_path=training_set[0], liver_path=training_set[1], batch_size=4):
                step += 1
                batch_x = batch_x_y[0]
                batch_y = batch_x_y[1]

                sess.run(train_step,
                         feed_dict={x_img: batch_x, y_true: batch_y})
                print("yes")


if __name__ == '__main__':
    train()
