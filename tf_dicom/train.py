#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/12 19:51
# @File    : train.py
# @Author  : NUS_LuoKe

from tf_dicom.load_dicom import *
import tensorflow as tf
from tf_dicom import dense_unet
import tensorlayer as tl

# base_dir = "/home/guest/notebooks/datasets/3Dircadb"
base_dir = "F:/IRCAD/3Dircadb1/"


def dice_coe(output, target, loss_type='jaccard', axis=(1, 2, 3), smooth=1e-5):
    inse = tf.reduce_sum(output * target, axis=axis)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output, axis=axis)
        r = tf.reduce_sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknow loss_type")

    dice = (2. * inse + smooth) / (l + r + smooth)
    dice = tf.reduce_mean(dice)
    return dice


def dice_hard_coe(output, target, threshold=0.5, axis=(1, 2, 3), smooth=1e-5):
    output = tf.cast(output > threshold, dtype=tf.float32)
    target = tf.cast(target > threshold, dtype=tf.float32)
    inse = tf.reduce_sum(tf.multiply(output, target), axis=axis)
    l = tf.reduce_sum(output, axis=axis)
    r = tf.reduce_sum(target, axis=axis)
    hard_dice = (2. * inse + smooth) / (l + r + smooth)
    return hard_dice


slice_path_list, liver_path_list = get_slice_liver_path(base_dir, shuffle=True)
training_set, validation_set, test_set = get_tra_val_test_set(slice_path_list, liver_path_list)

batch_size = 4
length = 512
width = 512
channel = 1


def train_and_val():
    # GPU limit
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.9

    # define placeholder
    x_img = tf.placeholder(tf.float32, shape=[batch_size, length, width, channel])
    y_true = tf.placeholder(tf.float32, shape=[batch_size, length, width, channel])

    # 1. Forward propagation
    pred = dense_unet.DenseNet(x_img, reduction=0.5)  # DenseNet_121(x_img, n_classes=3, is_train=True)
    y_pred = pred.outputs  # (4, 512, 512, 1)

    # 2. loss
    loss_ce = tf.reduce_mean(
        tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred), axis=(1, 2, 3)))
    loss_dice = 1 - dice_coe(tf.sigmoid(y_pred), y_true)
    loss = loss_dice + loss_ce

    # 3. dice
    sig_y_pred = tf.sigmoid(y_pred)
    dice = dice_hard_coe(sig_y_pred, y_true, threshold=0.5)

    # 4. optimizer
    learning_rate = 1e-3
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    with tf.Session(config=config) as sess:
        # initial  variables

        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # set epochs
        nb_epoch = 20
        for epoch in range(nb_epoch):
            print("EPOCH=%s:" % epoch)
            step = 0
            train_slice_path, train_liver_path = shuffle_parallel_list(training_set[0], training_set[1])
            val_slice_path, val_liver_path = shuffle_parallel_list(validation_set[0], validation_set[1])
            for train_batch_x_y in get_batch(slice_path=train_slice_path, liver_path=train_liver_path,
                                             batch_size=batch_size):
                step += 1
                train_batch_x = train_batch_x_y[0]
                train_batch_y = train_batch_x_y[1]
                step += 1

                _, train_loss, train_dice = sess.run([train_op, loss, dice],
                                                     feed_dict={x_img: train_batch_x, y_true: train_batch_y})

                print('Step %d, train loss = %.8f, train dice = %.8f' % (step, train_loss, train_dice))

                if step % 20 == 0:
                    for val_batch_x_y in get_batch(slice_path=val_slice_path, liver_path=val_liver_path,
                                                   batch_size=batch_size):
                        val_batch_x = val_batch_x_y[0]
                        val_batch_y = val_batch_x_y[1]
                        val_loss, val_dice = sess.run([loss, dice],
                                                      feed_dict={x_img: val_batch_x, y_true: val_batch_y})
                        print('Step %d, validation loss = %.8f, validation dice = %.8f' % (step, val_loss, val_dice))
                        break
        # testing
        print("finished training")
        print("*" * 30)
        print("*" * 30)
        test_slice_path, test_liver_path = shuffle_parallel_list(test_set[0], test_set[1])

        test_batch_x = []
        for image_path in test_slice_path:
            image_file = pydicom.dcmread(image_path)
            image_array = image_file.pixel_array
            test_batch_x.append(image_array)

        test_batch_y = []
        for image_path in test_liver_path:
            image_file = pydicom.dcmread(image_path)
            image_array = image_file.pixel_array
            test_batch_y.append(image_array)

        test_batch_x = np.asarray(test_batch_x)
        test_batch_x = test_batch_x.reshape((test_batch_x.shape[0], test_batch_x.shape[1], test_batch_x.shape[2], 1))
        test_batch_y = np.asarray(test_batch_y)
        test_batch_y = test_batch_y.reshape((test_batch_y.shape[0], test_batch_y.shape[1], test_batch_y.shape[2], 1))
        test_loss, test_dice = sess.run([loss, dice], feed_dict={x_img: test_batch_x, y_true: test_batch_y})
        print('test loss = %.8f, test dice = %.8f' % (test_loss, test_dice))


if __name__ == '__main__':
    train_and_val()
