from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# Interface to represent any agent
class Agent(object):

  def __init__(self, environment):
    pass

  def train(self):
    pass

  def play(self):
    pass

  def predict(self, state):
    pass
