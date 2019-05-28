# -*- coding: UTF-8 -*-
# Author: Liyou Wang
# Date:2018.7.4

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

count = 0


def save_net_to_npy(net):
    global count
    count += 1
    print(net.shape)
    np.save('./debug_npy_file/debug{}.npy'.format(count), net)
    return net


class Face_net(object):
    def __init__(self, inputs):
        self.inputs = inputs
        # self.weight_ini = tf.constant_initializer(w_ini)
        # self.bias_ini = tf.constant_initializer(b_ini)

    def face_net(self):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            ):
            net = slim.conv2d(self.inputs, num_outputs=32, kernel_size=[3, 3], stride=[2, 2],
                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                              padding='SAME', activation_fn=tf.nn.relu, scope='conv1a')  # 193 * 129
            net = slim.conv2d(net, num_outputs=32, kernel_size=[3, 3], stride=[1, 1],
                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                              padding='SAME', activation_fn=tf.nn.relu, scope='conv1b')   # 193 * 129
            net = slim.conv2d(net, num_outputs=48, kernel_size=[3, 3], stride=[2, 2],
                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                              padding='SAME', activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(0.001),
                              scope='conv2a')
            net = slim.conv2d(net, num_outputs=48, kernel_size=[3, 3], stride=[1, 1],
                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                              padding='SAME', activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(0.001),
                              scope='conv2b')  # 97 * 65
            net = slim.conv2d(net, num_outputs=72, kernel_size=[3, 3], stride=[2, 2],
                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                              padding='SAME', activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(0.001),
                              scope='conv3a')
            net = slim.conv2d(net, num_outputs=72, kernel_size=[3, 3], stride=[1, 1],
                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                              padding='SAME', activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(0.001),
                              scope='conv3b')  # 49 * 33
            net = slim.conv2d(net, num_outputs=108, kernel_size=[3, 3], stride=[2, 2],
                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                              padding='SAME', activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(0.001),
                              scope='conv4a')
            net = slim.conv2d(net, num_outputs=108, kernel_size=[3, 3], stride=[1, 1],
                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                              padding='SAME', activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(0.001),
                              scope='conv4b')  # 25 * 17
            # net = tf.py_func(save_net_to_npy, [net], tf.float32)
            # net.set_shape([None, 7, 7, 108])
            net = slim.conv2d(net, num_outputs=162, kernel_size=[3, 3], stride=[2, 2],
                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                              padding='SAME', activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(0.001),
                              scope='conv5a')
            net = slim.conv2d(net, num_outputs=162, kernel_size=[3, 3], stride=[1, 1],
                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                              padding='SAME', activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(0.001),
                              scope='conv5b')  # 13 * 9
            #net = slim.dropout(net, keep_prob=0.8)
            # net = slim.conv2d(net, num_outputs=243, kernel_size=[3, 3], stride=[2, 2],
            #                   weights_initializer=tf.contrib.layers.xavier_initializer(),
            #                   padding='SAME', activation_fn=tf.nn.relu, scope='conv6a')
            # net = slim.conv2d(net, num_outputs=243, kernel_size=[3, 3], stride=[1, 1],
            #                   weights_initializer=tf.contrib.layers.xavier_initializer(),
            #                   padding='SAME', activation_fn=tf.nn.relu, scope='conv6b')  # 4 * 4
            # net = slim.dropout(net, keep_prob=1.0)
            net = tf.reshape(net, [-1, 13 * 9 * 162])
            net = slim.fully_connected(net, 256, weights_initializer=tf.contrib.layers.xavier_initializer(),
                                       activation_fn=tf.nn.relu, scope='fc1')
            net = slim.fully_connected(net, 10090*3, weights_initializer=tf.contrib.layers.xavier_initializer(),
                                       activation_fn=None, scope='fc2')
            return net


