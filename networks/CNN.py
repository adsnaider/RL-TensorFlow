from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from .network import Network, conv2d, MLP


class CNN(Network):

  def __init__(self, name):
    super(CNN, self).__init__(self, name)

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

    with tf.variable_scope(name):
      self.inputs = tf.placeholder(
          shape=[None, history_size] + observation_dims,
          tf.float32,
          name='input')

      self.layers = [None] * (num_conv + 1 + 2)
      self.layers[0] = self.inputs
      for i in xrange(1, num_conv + 1):
        self.train_vars['w{}'.format(i)], self.train_vars['b{}'.format(
            i)], self.layers[i] = conv2d(
                self.layers[i - 1],
                kernel_dims,
                strides,
                depth,
                name='conv2d_{}'.format(i),
                trainable=trainable)

      self.train_vars['w{}'.format(num_conv + 1)], self.train_vars['b{}'.format(
          num_conv + 1)], self.layers[-2] = MLP(
              self.layers[-3],
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
