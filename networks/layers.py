from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def conv2d(x,
           kernel_dims,
           strides,
           depth,
           activation_fn=tf.nn.relu,
           weights_initializer=tf.contrib.layers.xavier_initializer,
           bias_initializer=tf.zeros_initializer,
           padding='VALID',
           name='conv2d',
           trainable=True):
  strides = [1, strides[0], strides[1], 1]
  kernel_shape = [kernel_dims[0], kernel_dims[1], x.get_shape()[-1], depth]
  with tf.variable_scope(name):
    w = tf.get_variable(
        'w',
        kernel_shape,
        tf.float32,
        initializer=weights_initializer,
        trainable=trainable)
    b = tf.get_variable(
        'b', [depth],
        tf.float32,
        initializer=bias_initializer,
        trainable=trainable)

    conv = tf.nn.conv2d(x, w, strides, padding, name='hidden_conv')
    out = tf.nn.bias_add(conv, b)
    if (activation_fn is not None):
      out = activation_fn(out)
    return w, b, out


def MLP(x,
        hidden_size,
        activation_fn=tf.nn.relu,
        weights_initializer=tf.contrib.layers.xavier_initializer,
        bias_initializer=tf.zeros_initializer,
        name='MLP',
        trainable=True):
  with tf.variable_scope(name):
    w = tf.get_variable(
        'w', [x.get_shape()[1], hidden_size],
        tf.float32,
        initializer=weights_initializer,
        trainable=trainable)
    b = tf.get_variable(
        'b', [hidden_size],
        tf.float32,
        initializer=bias_initializer,
        trainable=trainable)

    out = tf.matmul(x, w) + b
    if (activation_fn is not None):
      out = activation_fn(out)
    return w, b, out
