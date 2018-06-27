#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/12 19:51
# @File    : train.py
# @Author  : NUS_LuoKe

import tensorflow as tf

from tf_dicom import u_net
from tf_dicom.load_dicom import *

# base_dir = "F:/IRCAD/3Dircadb1/"
base_dir = "/home/guest/notebooks/datasets/3Dircadb"
checkpoint_dir = "/home/guest/notebooks/luoke/Model_Weights"

gpu_id = "0"
batch_size = 4
length = 512
width = 512
channel = 1
nb_epoch = 1000

# get training set from patient_1 to patient_18
train_patient_id_list = list(range(1, 19))
train_slice_path_list, train_liver_path_list = get_slice_liver_path(base_dir, patient_id_list=train_patient_id_list,
                                                                    shuffle=True)
training_set = [train_slice_path_list, train_liver_path_list]

# get validation set from patient_19
validation_slice_path_list, validation_liver_path_list = get_slice_liver_path(base_dir, patient_id_list=[19],
                                                                              shuffle=True)
validation_set = [validation_slice_path_list, validation_liver_path_list]

# get test set from patient_20
test_slice_path_list, test_liver_path_list = get_slice_liver_path(base_dir, patient_id_list=[20], shuffle=True)
test_set = [test_slice_path_list, test_liver_path_list]

# default color_dict for label 1-6
default_color_dict = [
    {'label': 1, 'color': [200, None, None]},
    {'label': 2, 'color': [None, 200, None]},
    {'label': 3, 'color': [None, None, 200]},
    {'label': 4, 'color': [200, 200, None]},
    {'label': 5, 'color': [None, 200, 200]},
    {'label': 6, 'color': [200, None, 200]},
]


def display_segment(image, label, color_dicts=default_color_dict):
    """Display segmentation results on original image.

    Params
    ------
        image       : np.array(uint8): an array with shape (w,h)|(w,h,1)|(w,h,c) for original image
        label       : np.array(int)  : an int(uint) array with shape (w,h)|(w,h,1) for label image
        color_dicts : list of dicts  : a list of dictionary include label index and color

    Returns
    -------
        image : np.array : an array for image with label

    Examples
    --------
    >>> display = display_segment(image, label)
    >>> display = display_segment(image, label, color_dicts=[{'label':1, 'color':[255,255,255]}])
    """
    if image.ndim == 2:
        image = np.repeat(np.expand_dims(image, axis=-1), 3, axis=2)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)

    for cd in color_dicts:
        for i, c in enumerate(cd['color']):
            if c is not None:
                image[:, :, i:i + 1][label == cd['label']] = c
    return image


def display_batch_segment(images, labels, color_dicts=default_color_dict):
    """Display segmentation results on a batch of original images.

    Params
    ------
        images      : np.array(uint8): an array with shape (b,w,h)|(b,w,h,1)|(b,w,h,c) for original images batch
        labels      : np.array(int)  : an int(uint) array with shape (b,w,h)|(b,w,h,1) for label images batch
        color_dicts : list of dicts  : a list of dictionary include label index and color

    Returns
    -------
        images : np.array : an array for a batch of image with label

    Examples
    --------
    >>> display = display_segment(images, labels)
    >>> display = display_segment(images, labels, color_dicts=[{'label':1, 'color':[255,255,255]}])
    """
    if images.ndim == 3:
        images = np.repeat(np.expand_dims(images, axis=-1), 3, axis=3)
    elif images.ndim == 4 and images.shape[3] == 1:
        images = np.repeat(images, 3, axis=3)

    for cd in color_dicts:
        for i, c in enumerate(cd['color']):
            if c is not None:
                images[:, :, :, i:i + 1][labels == cd['label']] = c
    return images


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
    # hard_dice = tf.reduce_mean(hard_dice)
    return hard_dice


def train_and_val(gpu_id):
    # GPU limit
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.9

    # define placeholder
    x_img = tf.placeholder(tf.float32, shape=[batch_size, length, width, channel])
    y_true = tf.placeholder(tf.float32, shape=[batch_size, length, width, channel])

    # 1. Forward propagation
    pred = u_net.DenseNet(x_img, reduction=0.5)  # DenseNet_121(x_img, n_classes=3, is_train=True)
    y_pred = pred.outputs  # (4, 512, 512, 1)

    # 2. loss
    loss_ce = tf.reduce_mean(
        tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred), axis=(1, 2, 3)))
    loss_dice = 1 - dice_coe(tf.sigmoid(y_pred), y_true)
    # loss = loss_dice + loss_ce
    loss = loss_dice

    # 3. dice
    sig_y_pred = tf.sigmoid(y_pred)
    dice = dice_hard_coe(sig_y_pred, y_true, threshold=0.5)

    # 4. optimizer
    learning_rate = 1e-4
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    # train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # saver
    saver = tf.train.Saver()

    # define init_op
    init_op = tf.global_variables_initializer()

    with tf.Session(config=config) as sess:
        print("start session...")
        print("The total number of training epoch is: %s " % nb_epoch)

        # initial  variables
        sess.run(init_op)

        step = 0
        for epoch in range(nb_epoch):
            print("EPOCH=%s:" % epoch)
            train_slice_path, train_liver_path = shuffle_parallel_list(training_set[0], training_set[1])
            val_slice_path, val_liver_path = shuffle_parallel_list(validation_set[0], validation_set[1])
            for train_batch_x_y in get_batch_crop_center(train_slice_path, train_liver_path, batch_size=batch_size,
                                                         crop_by_center=False):
                step += 1
                train_batch_x = train_batch_x_y[0]
                train_batch_y = train_batch_x_y[1]
                train_batch_x, train_batch_y = enlarge_slice(train_batch_x, train_batch_y, batch_size=batch_size,
                                                             length=length, width=width)

                _, train_loss, train_dice, _y_true = sess.run([train_op, loss, dice, y_true],
                                                              feed_dict={x_img: train_batch_x, y_true: train_batch_y})

                # tl.vis.save_images(train_batch_x, [2, 2], '/home/guest/notebooks/luoke/vis/ori_{}.png'.format(step))
                # display = display_batch_segment(train_batch_x, train_batch_y)
                # tl.vis.save_images(display, [2, 2], '/home/guest/notebooks/luoke/vis/seg_{}.png'.format(step))

                if step % 5 == 0:
                    print('Step %d, train loss = %.8f, train dice = %.8f' % (
                        step, train_loss, np.mean(train_dice[np.sum(_y_true, axis=(1, 2, 3)) > 0])))

                if step % 200 == 0:
                    for val_batch_x_y in get_batch_crop_center(val_slice_path, val_liver_path, batch_size=batch_size,
                                                               crop_by_center=False):
                        val_batch_x = val_batch_x_y[0]
                        val_batch_y = val_batch_x_y[1]
                        val_batch_x, val_batch_y = enlarge_slice(val_batch_x, val_batch_y, batch_size=batch_size,
                                                                 length=length, width=width)

                        val_loss, val_dice, _y_true = sess.run([loss, dice, y_true],
                                                               feed_dict={x_img: val_batch_x, y_true: val_batch_y})
                        print('Step %d, validation loss = %.8f, validation dice = %.8f' % (
                            step, val_loss, np.mean(val_dice[np.sum(_y_true, axis=(1, 2, 3)) > 0])))
                        print("\n")
                        break
            if step % 5000 == 0:
                saver.save(sess, checkpoint_dir + 'model.ckpt', global_step=step)

        print("finished training")
        print("begin to test...")
        test_slice_path, test_liver_path = shuffle_parallel_list(test_set[0], test_set[1])
        count = 0
        test_batch_size = 20
        for test_batch_x_y in get_batch_crop_center(test_slice_path, test_liver_path, batch_size=test_batch_size,
                                                    crop_by_center=False):
            count += 1
            test_batch_x = test_batch_x_y[0]
            test_batch_y = test_batch_x_y[1]
            test_batch_x, test_batch_y = enlarge_slice(test_batch_x, test_batch_y, batch_size=test_batch_size,
                                                       length=length, width=width)
            test_loss, test_dice, _y_true = sess.run([loss, dice, y_true],
                                                     feed_dict={x_img: test_batch_x, y_true: test_batch_y})
            print('test loss = %.8f, test dice = %.8f' % (
                test_loss, np.mean(test_dice[np.sum(_y_true, axis=(1, 2, 3)) > 0])))
            print("\n")
            if count == 5:
                break

        print("*" * 30)
        print("*" * 30)


if __name__ == '__main__':
    train_and_val(gpu_id)
