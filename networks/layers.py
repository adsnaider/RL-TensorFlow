from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def conv2d(x,
           kernel_dims,
           strides,
           depth,
           activation_fn=tf.nn.relu,
           weights_initializer=tf.contrib.layers.xavier_initializer(),
           bias_initializer=tf.zeros_initializer,
           padding='VALID',
           name='conv2d',
           trainable=True):
  strides = [1, strides[0], strides[1], 1]
  kernel_shape = [kernel_dims[0], kernel_dims[1], x.get_shape()[-1], depth]
  with tf.variable_scope(name, reuse=False):
    w = tf.get_variable(
        name='w',
        shape=kernel_shape,
        dtype=tf.float32,
        initializer=weights_initializer,
        trainable=trainable)
    b = tf.get_variable(
        name='b',
        shape=[depth],
        dtype=tf.float32,
        initializer=bias_initializer,
        trainable=trainable)

    conv = tf.nn.conv2d(x, w, strides, padding, name='hidden_conv')
    out = tf.nn.bias_add(conv, b)
    if (activation_fn is not None):
      out = activation_fn(out)
    return w, b, out


def MLP(x,
        size,
        activation_fn=tf.nn.relu,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        bias_initializer=tf.zeros_initializer,
        name='MLP',
        trainable=True):
  with tf.variable_scope(name, reuse=False):
    w = tf.get_variable(
        name='w',
        shape=[x.get_shape()[1], size],
        dtype=tf.float32,
        initializer=weights_initializer,
        trainable=trainable)
    b = tf.get_variable(
        name='b',
        shape=[size],
        dtype=tf.float32,
        initializer=bias_initializer,
        trainable=trainable)

    out = tf.matmul(x, w) + b
    if (activation_fn is not None):
      out = activation_fn(out)
    return w, b, out
