#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/12 19:40
# @File    : dense_unet.py
# @Author  : NUS_LuoKe


import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import BatchNormLayer, MaxPool2d, DropoutLayer, ConcatLayer
from tensorlayer.layers import Conv2dLayer
from tensorlayer.layers import UpSampling2dLayer, MeanPool2d

is_train = True


def conv_block(x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
    '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
        # Arguments
            x: input tensor
            stage: index for dense block
            branch: layer index within each dense block
            nb_filter: number of filters
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''
    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    batch_size, x_row, x_col, x_channels = x.outputs.get_shape().as_list()
    # 1x1 Convolution (Bottleneck layer)
    inter_channel = nb_filter * 4
    x = BatchNormLayer(x, epsilon=eps, act=tf.nn.relu, is_train=is_train,
                       name=conv_name_base + '_x1_bn')
    x = Conv2dLayer(x, shape=[1, 1, x_channels, inter_channel], strides=[1, 1, 1, 1],
                    b_init=None, name=conv_name_base + '_x1')

    if dropout_rate:
        x = DropoutLayer(x, keep=dropout_rate)

    # 3x3 Convolution
    x = BatchNormLayer(x, epsilon=eps, act=tf.nn.relu, is_train=is_train,
                       name=conv_name_base + '_x2_bn')
    x_channels = x.outputs.get_shape().as_list()[3]
    x = Conv2dLayer(x, shape=[3, 3, x_channels, nb_filter], strides=[1, 1, 1, 1], padding='SAME',
                    b_init=None, name=conv_name_base + '_x2')
    if dropout_rate:
        x = DropoutLayer(x, keep=dropout_rate)
    return x


def dense_bolck(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None,
                weight_decay=1e-4, grow_nb_filters=True):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_layers: the number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''
    eps = 1.1e-5
    concat_feat = x
    for i in range(nb_layers):
        branch = i + 1
        x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)
        concat_feat = ConcatLayer([concat_feat, x], concat_dim=concat_axis,
                                  name='concat_' + str(stage) + '_' + str(branch))

        if grow_nb_filters:
            nb_filter += growth_rate
    return concat_feat, nb_filter


def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_filter: number of filters
            compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''
    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage)

    batch_size, x_row, x_col, x_channels = x.outputs.get_shape().as_list()

    x = BatchNormLayer(x, act=tf.nn.relu, epsilon=eps, is_train=is_train,
                       name=conv_name_base + '_bn')
    x = Conv2dLayer(x, shape=[1, 1, x_channels, nb_filter * compression], strides=[1, 1, 1, 1],
                    name=conv_name_base, b_init=None)
    if dropout_rate:
        x = DropoutLayer(x, keep=dropout_rate)

    x = MeanPool2d(x, filter_size=[2, 2], strides=[2, 2], name=pool_name_base)

    return x


def DenseNet(x, nb_dense_block=4, growth_rate=48, nb_filter=96, reduction=0.0,
             dropout_rate=0.0, weight_decay=1e-4, classes=1000, weights_path=None):
    '''Instantiate the DenseNet 161 architecture,
        # Arguments
            x:input data (512*512*1)
            nb_dense_block: number of dense blocks to add to end
            growth_rate: number of filters to add per dense block
            nb_filter: initial number of filters
            reduction: reduction factor of transition blocks.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            classes: optional number of classes to classify images
            weights_path: path to pre-trained weights
        # Returns
            x.shape = (batch_size, 512, 512, 1)
            call x.output, then it is a 4D tensor
    '''
    # From architecture for ImageNet (Table 1 in the paper)
    global concat_axis
    concat_axis = 3
    eps = 1.1e-5
    compression = 1.0 - reduction
    nb_filter = 96
    nb_layers = [6, 12, 36, 24]  # For DenseNet-161
    batch_size, x_row, x_col, x_channels = x.get_shape().as_list()
    # tensorlayer data format
    inputs = tl.layers.InputLayer(x, name='input_layer')

    # Initial convolution
    x = Conv2dLayer(inputs, shape=[7, 7, x_channels, nb_filter], strides=[1, 2, 2, 1], padding='SAME',
                    b_init=None, name='cov1_1')
    x = BatchNormLayer(x, act=tf.nn.relu, epsilon=eps, is_train=is_train, name='bn1')
    x = MaxPool2d(x, filter_size=[3, 3], strides=[2, 2], name='maxpool_1')
    #    print(x.outputs.get_shape().as_list())

    # Add dense block
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx + 2
        x, nb_filter = dense_bolck(x, stage, nb_layers[block_idx], nb_filter, growth_rate,
                                   dropout_rate=dropout_rate,
                                   weight_decay=weight_decay)

        # Add transition_block
        x = transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate,
                             weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)
    #        print(x.outputs.get_shape().as_list())

    # Last dens_block
    final_stage = stage + 1
    x, nb_filter = dense_bolck(x, final_stage, nb_layers[-1], nb_filter, growth_rate,
                               dropout_rate=dropout_rate, weight_decay=weight_decay)

    x = BatchNormLayer(x, act=tf.nn.relu, epsilon=eps, is_train=is_train, name='conv' + str(final_stage) + '_blk_bn')
    #    print(x.outputs.get_shape().as_list())

    # upsampling and conv (0)
    up0 = UpSampling2dLayer(x, size=[2, 2], name='up0')
    up0_channels = up0.outputs.get_shape().as_list()[3]
    conv_up0 = Conv2dLayer(up0, shape=[3, 3, up0_channels, 768], strides=[1, 1, 1, 1], padding='SAME',
                           b_init=None, name="conv_up0")
    ac_up0 = BatchNormLayer(conv_up0, act=tf.nn.relu, is_train=is_train, name='ac_up0')
    #    print(ac_up0.outputs.get_shape().as_list())

    # upsampling and conv (1)
    up1 = UpSampling2dLayer(ac_up0, size=[2, 2], name='up1')
    up1_chanels = up1.outputs.get_shape().as_list()[3]
    conv_up1 = Conv2dLayer(up1, shape=[3, 3, up1_chanels, 384], strides=[1, 1, 1, 1], padding='SAME',
                           b_init=None, name="conv_up1")
    ac_up1 = BatchNormLayer(conv_up1, act=tf.nn.relu, is_train=is_train, name='ac_up1')
    #    print(ac_up1.outputs.get_shape().as_list())

    # upsampling and conv (2)
    up2 = UpSampling2dLayer(ac_up1, size=[2, 2], name='up2')
    up2_chanels = up2.outputs.get_shape().as_list()[3]
    conv_up2 = Conv2dLayer(up2, shape=[3, 3, up2_chanels, 96], strides=[1, 1, 1, 1], padding='SAME',
                           b_init=None, name="conv_up2")
    ac_up2 = BatchNormLayer(conv_up2, act=tf.nn.relu, is_train=is_train, name='ac_up2')
    #    print(ac_up2.outputs.get_shape().as_list())

    # upsampling and conv (3)
    up3 = UpSampling2dLayer(ac_up2, size=[2, 2], name='up3')
    up3_chanels = up3.outputs.get_shape().as_list()[3]
    conv_up3 = Conv2dLayer(up3, shape=[3, 3, up3_chanels, 96], strides=[1, 1, 1, 1], padding='SAME',
                           b_init=None, name="conv_up3")
    ac_up3 = BatchNormLayer(conv_up3, act=tf.nn.relu, is_train=is_train, name='ac_up3')
    #    print(ac_up3.outputs.get_shape().as_list())

    # upsampling and conv (4)
    up4 = UpSampling2dLayer(ac_up3, size=[2, 2], name='up4')
    up4_chanels = up4.outputs.get_shape().as_list()[3]
    conv_up4 = Conv2dLayer(up4, shape=[3, 3, up4_chanels, 64], strides=[1, 1, 1, 1], padding='SAME',
                           b_init=None, name="conv_up4")
    ac_up4 = BatchNormLayer(conv_up4, act=tf.nn.relu, is_train=is_train, name='ac_up4')
    #    print(ac_up4.outputs.get_shape().as_list())

    # Last convolution
    ac_up4_chanels = ac_up4.outputs.get_shape().as_list()[3]
    x = Conv2dLayer(ac_up4, shape=[1, 1, ac_up4_chanels, 1], strides=[1, 1, 1, 1], padding='SAME',
                    b_init=None, name="dense167classifer")

    return x


