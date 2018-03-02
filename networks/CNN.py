from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from networks.network import Network
from networks.layers import conv2d, MLP


class CNN(Network):

  def __init__(self, name):
    super(CNN, self).__init__(name)

  def build_output_op(self,
                      depth,
                      hidden_size,
                      output_size,
                      num_conv,
                      kernel_dims,
                      strides,
                      observation_dims,
                      history_size,
                      hidden_activation_fn=tf.nn.relu,
                      output_activation_fn=None,
                      trainable=True):
    if (self.inputs is not None):
      raise Exception('build_output_op must be called exactly once')

    with tf.variable_scope(self.name) as self.main_scope:
      self.inputs = tf.placeholder(
          shape=[None, history_size] + observation_dims,
          dtype=tf.float32,
          name='input')

      reshape = tf.reshape(
          self.inputs,
          [-1, observation_dims[0] * history_size] + observation_dims[1:])

      self.layers = [None] * (num_conv + 1 + 2)
      self.layers[0] = reshape
      for i in xrange(1, num_conv + 1):
        self.train_vars['w{}'.format(i)], self.train_vars['b{}'.format(
            i)], self.layers[i] = conv2d(
                self.layers[i - 1],
                kernel_dims,
                strides,
                depth,
                name='conv2d_{}'.format(i),
                trainable=trainable)

      shape = self.layers[-3].get_shape().as_list()
      reshape = tf.reshape(self.layers[-3],
                           [-1, shape[1] * shape[2] * shape[3]])

      self.train_vars['w{}'.format(num_conv + 1)], self.train_vars['b{}'.format(
          num_conv + 1)], self.layers[-2] = MLP(
              reshape,
              hidden_size,
              hidden_activation_fn,
              name='hidden_MLP',
              trainable=trainable)

      self.train_vars['w_out'], self.train_vars['b_out'], self.layers[-1] = MLP(
          self.layers[-2],
          output_size,
          output_activation_fn,
          name='out',
          trainable=trainable)

      self.outputs = self.layers[-1]
      self.actions = tf.argmax(self.outputs, axis=1, name='actions')
