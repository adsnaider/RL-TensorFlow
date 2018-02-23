from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


# Interface to represent any neural network
class Network(object):

  def __init__(self, name):
    self.name = name
    self.train_vars = {}
    self.copy_op = None
    self.actions = None
    self.outputs = None
    self.inputs = None
    self.main_scope = None

  def build_output_op(self, input_layer, hidden_sizes, output_size,
                      weights_initializer, bias_initializer,
                      hidden_activation_fn, output_activation_fn, trainable):
    raise NotImplementedError

  def create_copy_op(self, target_network):
    assert (set(self.train_vars.keys()) == set(
        target_network.train_vars.keys()))

    with tf.variable_scope(self.main_scope):
      with tf.variable_scope('copy'):
        copy_ops = []
        for k, v in target_network.train_vars.iteritems():
          copy_ops.append(
              tf.assign(self.train_vars[k], v, name='copy_assign_' + k))

        self.copy_op = tf.group(copy_ops, name='copy_op')

  def run_copy_op(self, sess):
    if (self.copy_op == None):
      raise Exception('create_copy_op must be called before run_copy_op')
    sess.run(self.copy_op)

  def calc_actions(self, observation, sess):
    if (self.actions == None):
      raise Exception('build_output_op must be called before calc_actions')
    return sess.run(self.actions, feed_dict={self.inputs: observation})

  def calc_outputs(self, observation, sess):
    if (self.outputs == None):
      raise Exception('build_output_op must be called before calc_actions')
    return sess.run(self.outputs, feed_dict={self.inputs: observation})
