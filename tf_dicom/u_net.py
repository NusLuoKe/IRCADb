#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/22 11:56
# @File    : u_net.py
# @Author  : NUS_LuoKe


import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

is_train = True


def u_net(x, reuse=False, n_out=1):
    _, nx, ny, nz = x.get_shape().as_list()
    with tf.variable_scope("u_net", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        inputs = InputLayer(x, name='inputs')
        conv1 = Conv2d(inputs, 64, (3, 3), act=tf.nn.relu, name='conv1_1')
        conv1 = Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, name='conv1_2')
        pool1 = MaxPool2d(conv1, (2, 2), name='pool1')
        conv2 = Conv2d(pool1, 128, (3, 3), act=tf.nn.relu, name='conv2_1')
        conv2 = Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, name='conv2_2')
        pool2 = MaxPool2d(conv2, (2, 2), name='pool2')
        conv3 = Conv2d(pool2, 256, (3, 3), act=tf.nn.relu, name='conv3_1')
        conv3 = Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, name='conv3_2')
        pool3 = MaxPool2d(conv3, (2, 2), name='pool3')
        conv4 = Conv2d(pool3, 512, (3, 3), act=tf.nn.relu, name='conv4_1')
        conv4 = Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, name='conv4_2')
        pool4 = MaxPool2d(conv4, (2, 2), name='pool4')
        conv5 = Conv2d(pool4, 1024, (3, 3), act=tf.nn.relu, name='conv5_1')
        conv5 = Conv2d(conv5, 1024, (3, 3), act=tf.nn.relu, name='conv5_2')

        up4 = DeConv2d(conv5, 512, (3, 3), (nx / 8, ny / 8), (2, 2), name='deconv4')
        up4 = ConcatLayer([up4, conv4], 3, name='concat4')
        conv4 = Conv2d(up4, 512, (3, 3), act=tf.nn.relu, name='uconv4_1')
        conv4 = Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, name='uconv4_2')
        up3 = DeConv2d(conv4, 256, (3, 3), (nx / 4, ny / 4), (2, 2), name='deconv3')
        up3 = ConcatLayer([up3, conv3], 3, name='concat3')
        conv3 = Conv2d(up3, 256, (3, 3), act=tf.nn.relu, name='uconv3_1')
        conv3 = Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, name='uconv3_2')
        up2 = DeConv2d(conv3, 128, (3, 3), (nx / 2, ny / 2), (2, 2), name='deconv2')
        up2 = ConcatLayer([up2, conv2], 3, name='concat2')
        conv2 = Conv2d(up2, 128, (3, 3), act=tf.nn.relu, name='uconv2_1')
        conv2 = Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, name='uconv2_2')
        up1 = DeConv2d(conv2, 64, (3, 3), (nx / 1, ny / 1), (2, 2), name='deconv1')
        up1 = ConcatLayer([up1, conv1], 3, name='concat1')
        conv1 = Conv2d(up1, 64, (3, 3), act=tf.nn.relu, name='uconv1_1')
        conv1 = Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, name='uconv1_2')
        conv1 = Conv2d(conv1, n_out, (1, 1), act=tf.nn.sigmoid, name='uconv1')
    return conv1


def u_net_bn(x, is_train=False, reuse=False, batch_size=None, pad='SAME', n_out=1):
    """image to image translation via conditional adversarial learning"""
    nx = int(x._shape[1])
    ny = int(x._shape[2])
    nz = int(x._shape[3])
    print(" * Input: size of image: %d %d %d" % (nx, ny, nz))

    w_init = tf.truncated_normal_initializer(stddev=0.01)
    b_init = tf.constant_initializer(value=0.0)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("u_net", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        inputs = InputLayer(x, name='inputs')

        conv1 = Conv2d(inputs, 64, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv1')
        conv2 = Conv2d(conv1, 128, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv2')
        conv2 = BatchNormLayer(conv2, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init,
                               name='bn2')

        conv3 = Conv2d(conv2, 256, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv3')
        conv3 = BatchNormLayer(conv3, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init,
                               name='bn3')

        conv4 = Conv2d(conv3, 512, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv4')
        conv4 = BatchNormLayer(conv4, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init,
                               name='bn4')

        conv5 = Conv2d(conv4, 512, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv5')
        conv5 = BatchNormLayer(conv5, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init,
                               name='bn5')

        conv6 = Conv2d(conv5, 512, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv6')
        conv6 = BatchNormLayer(conv6, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init,
                               name='bn6')

        conv7 = Conv2d(conv6, 512, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv7')
        conv7 = BatchNormLayer(conv7, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init,
                               name='bn7')

        conv8 = Conv2d(conv7, 512, (4, 4), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2), padding=pad, W_init=w_init,
                       b_init=b_init, name='conv8')
        print(" * After conv: %s" % conv8.outputs)
        # exit()
        # print(nx/8)
        up7 = DeConv2d(conv8, 512, (4, 4), out_size=(2, 2), strides=(2, 2),
                       padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv7')
        up7 = BatchNormLayer(up7, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn7')

        # print(up6.outputs)
        up6 = ConcatLayer([up7, conv7], concat_dim=3, name='concat6')
        up6 = DeConv2d(up6, 1024, (4, 4), out_size=(4, 4), strides=(2, 2),
                       padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv6')
        up6 = BatchNormLayer(up6, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn6')
        # print(up6.outputs)
        # exit()

        up5 = ConcatLayer([up6, conv6], concat_dim=3, name='concat5')
        up5 = DeConv2d(up5, 1024, (4, 4), out_size=(8, 8), strides=(2, 2),
                       padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv5')
        up5 = BatchNormLayer(up5, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn5')
        # print(up5.outputs)
        # exit()

        up4 = ConcatLayer([up5, conv5], concat_dim=3, name='concat4')
        up4 = DeConv2d(up4, 1024, (4, 4), out_size=(15, 15), strides=(2, 2),
                       padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv4')
        up4 = BatchNormLayer(up4, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn4')

        up3 = ConcatLayer([up4, conv4], concat_dim=3, name='concat3')
        up3 = DeConv2d(up3, 256, (4, 4), out_size=(30, 30), strides=(2, 2),
                       padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv3')
        up3 = BatchNormLayer(up3, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn3')

        up2 = ConcatLayer([up3, conv3], concat_dim=3, name='concat2')
        up2 = DeConv2d(up2, 128, (4, 4), out_size=(60, 60), strides=(2, 2),
                       padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv2')
        up2 = BatchNormLayer(up2, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn2')

        up1 = ConcatLayer([up2, conv2], concat_dim=3, name='concat1')
        up1 = DeConv2d(up1, 64, (4, 4), out_size=(120, 120), strides=(2, 2),
                       padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv1')
        up1 = BatchNormLayer(up1, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn1')

        up0 = ConcatLayer([up1, conv1], concat_dim=3, name='concat0')
        up0 = DeConv2d(up0, 64, (4, 4), out_size=(240, 240), strides=(2, 2),
                       padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv0')
        up0 = BatchNormLayer(up0, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn0')
        # print(up0.outputs)
        # exit()

        out = Conv2d(up0, n_out, (1, 1), act=tf.nn.sigmoid, name='out')

        print(" * Output: %s" % out.outputs)
        # exit()

    return out


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


def DenseNet(x, nb_dense_block=4, growth_rate=48, reduction=0.0,
             dropout_rate=0.0, weight_decay=1e-4):
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

    # Last dens_block
    final_stage = stage + 1
    x, nb_filter = dense_bolck(x, final_stage, nb_layers[-1], nb_filter, growth_rate,
                               dropout_rate=dropout_rate, weight_decay=weight_decay)

    x = BatchNormLayer(x, act=tf.nn.relu, epsilon=eps, is_train=is_train, name='conv' + str(final_stage) + '_blk_bn')

    # upsampling and conv (0)
    up0 = UpSampling2dLayer(x, size=[2, 2], name='up0')
    up0_channels = up0.outputs.get_shape().as_list()[3]
    conv_up0 = Conv2dLayer(up0, shape=[3, 3, up0_channels, 768], strides=[1, 1, 1, 1], padding='SAME',
                           b_init=None, name="conv_up0")
    ac_up0 = BatchNormLayer(conv_up0, act=tf.nn.relu, is_train=is_train, name='ac_up0')

    # upsampling and conv (1)
    up1 = UpSampling2dLayer(ac_up0, size=[2, 2], name='up1')
    up1_chanels = up1.outputs.get_shape().as_list()[3]
    conv_up1 = Conv2dLayer(up1, shape=[3, 3, up1_chanels, 384], strides=[1, 1, 1, 1], padding='SAME',
                           b_init=None, name="conv_up1")
    ac_up1 = BatchNormLayer(conv_up1, act=tf.nn.relu, is_train=is_train, name='ac_up1')

    # upsampling and conv (2)
    up2 = UpSampling2dLayer(ac_up1, size=[2, 2], name='up2')
    up2_chanels = up2.outputs.get_shape().as_list()[3]
    conv_up2 = Conv2dLayer(up2, shape=[3, 3, up2_chanels, 96], strides=[1, 1, 1, 1], padding='SAME',
                           b_init=None, name="conv_up2")
    ac_up2 = BatchNormLayer(conv_up2, act=tf.nn.relu, is_train=is_train, name='ac_up2')

    # upsampling and conv (3)
    up3 = UpSampling2dLayer(ac_up2, size=[2, 2], name='up3')
    up3_chanels = up3.outputs.get_shape().as_list()[3]
    conv_up3 = Conv2dLayer(up3, shape=[3, 3, up3_chanels, 96], strides=[1, 1, 1, 1], padding='SAME',
                           b_init=None, name="conv_up3")
    ac_up3 = BatchNormLayer(conv_up3, act=tf.nn.relu, is_train=is_train, name='ac_up3')

    # upsampling and conv (4)
    up4 = UpSampling2dLayer(ac_up3, size=[2, 2], name='up4')
    up4_chanels = up4.outputs.get_shape().as_list()[3]
    conv_up4 = Conv2dLayer(up4, shape=[3, 3, up4_chanels, 64], strides=[1, 1, 1, 1], padding='SAME',
                           b_init=None, name="conv_up4")
    ac_up4 = BatchNormLayer(conv_up4, act=tf.nn.relu, is_train=is_train, name='ac_up4')

    # Last convolution
    ac_up4_chanels = ac_up4.outputs.get_shape().as_list()[3]
    x = Conv2dLayer(ac_up4, shape=[1, 1, ac_up4_chanels, 1], strides=[1, 1, 1, 1], padding='SAME',
                    b_init=None, name="dense167classifer")

    return x
