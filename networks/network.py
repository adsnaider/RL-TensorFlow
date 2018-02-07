from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# Interface to represent any neural network
class Network(object):

  def __init__(self, sess, name=None):
    raise NotImplementedError

  def build_output_op(self, input_layer, hidden_sizes, output_size,
                      weights_initializer, bias_initializer,
                      hidden_activation_fn, output_activation_fn, trainable):
    raise NotImplementedError

  def create_copy_op(self, network_target):
    raise NotImplementedError

  def run_copy_op(self):
    raise NotImplementedError

  def calc_actions(self, observation):
    raise NotImplementedError

  def calc_outputs(self, observation):
    raise NotImplementedError
