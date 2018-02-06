from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# Interface to represent any neural network
class Network(object):

  def __init__(self, sess, name=None):
    pass

  def build_output_op(self, input_layer, hidden_sizes, output_size,
                      weights_initializer, bias_initializer,
                      hidden_activation_fn, output_activation_fn, trainable):
    pass

  def create_copy_op(self, network_target):
    pass

  def run_copy_op(self):
    pass

  def calc_actions(self, observation):
    pass

  def calc_outputs(self, observation):
    pass
