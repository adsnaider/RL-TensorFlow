from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from networks.network import Network
from networks.layers import conv2d, MLP


class ColorFightNetwork(Network):

  def __init__(self, name):
    super(ColorFightNetwork, self).__init__(name)
    self.cnn_inputs = None
    self.fc_inputs = None

  def build_output_op(self,
                      grid_shape,
                      cnn_input_dims,
                      fc_input_size,
                      kernel_dims,
                      depth,
                      hidden_size,
                      num_conv,
                      strides,
                      hidden_activation_fn=tf.nn.relu,
                      actor=True,
                      trainable=True):
    self.grid_shape = grid_shape
    if (self.cnn_inputs is not None and
        self.fully_connected_inputs is not None):
      raise Exception('build_output_op must be called exactly once')
    if (cnn_input_dims[0] != grid_shape[0] or
        cnn_input_dims[1] != grid_shape[1]):
      raise Exception('grid shape must match cnn shape')

    with tf.variable_scope(self.name) as self.main_scope:
      self.cnn_inputs = tf.placeholder(
          shape=[None] + cnn_input_dims, dtype=tf.float32, name='cnn_input')

      self.layers = [None] * (num_conv + 1 + 2)
      self.layers[0] = self.cnn_inputs
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

      self.fc_inputs = tf.placeholder(
          shape=[None, fc_input_size], dtype=tf.float32, name='fc_input')
      concat = tf.concat([reshape, self.fc_inputs], axis=1)

      self.train_vars['w{}'.format(num_conv + 1)], self.train_vars['b{}'.format(
          num_conv + 1)], self.layers[-2] = MLP(
              concat,
              hidden_size,
              hidden_activation_fn,
              name='hidden_MLP',
              trainable=trainable)

      if (actor):

        self.output_layers = [None] * 4
        self.train_vars['w_out_loc'], self.train_vars[
            'b_out_loc'], self.output_layers[0] = MLP(
                self.layers[-2],
                grid_shape[0] * grid_shape[1],
                tf.nn.softmax,
                name='out_loc',
                trainable=trainable)

        self.train_vars['w_out_type'], self.train_vars[
            'b_out_type'], self.output_layers[1] = MLP(
                self.layers[-2],
                3,
                tf.nn.softmax,
                name='out_type',
                trainable=trainable)

        self.train_vars['w_out_blast_dir'], self.train_vars[
            'b_out_blast_dir'], self.output_layers[2] = MLP(
                self.layers[-2],
                3,
                tf.nn.softmax,
                name='out_blast_dir',
                trainable=trainable)

        self.train_vars['w_out_blast_type'], self.train_vars[
            'b_out_blast_type'], self.output_layers[3] = MLP(
                self.layers[-2],
                2,
                tf.nn.softmax,
                name='out_blast_type',
                trainable=trainable)

        self.layers[-1] = tf.concat(self.output_layers, axis=1)
      else:
        self.train_vars['w_out'], self.train_vars['b_out'], self.layers[
            -1] = MLP(
                self.layers[-2], 1, None, name='out', trainable=trainable)

      self.outputs = self.layers[-1]
      self.actions = None

  def calc_actions(self, grid_input, extra_inputs, sess):
    if (self.outputs == None):
      raise Exception('build_output_op must be called before calc_actions')
    outs = sess.run(
        self.output_layers,
        feed_dict={
            self.cnn_inputs: grid_input,
            self.fc_inputs: extra_inputs
        })

    loc = np.argmax(outs[0], axis=1)
    loc = np.array([loc // grid_shape[0], loc % grid_shape[0]])

    action_type = np.argmax(outs[1], axis=1)

    blast_dir = np.argmax(outs[2], axis=1)
    blast_type = np.argmax(outs[3], axis=1)

    return loc, action_type, blast_dir, blast_type

  def calc_outputs(self, grid_input, extra_inputs, sess):
    if (self.outputs == None):
      raise Exception('build_output_op must be called before calc_actions')
    return sess.run(
        self.outputs,
        feed_dict={
            self.cnn_inputs: grid_input,
            self.fc_inputs: extra_inputs
        })
