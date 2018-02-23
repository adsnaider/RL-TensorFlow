from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# Interface to represent any environment.
# It contains the the game mechanics
class Environment(object):

  def __init__(self):
    self.actions = None
    self.view = None

  def new_game(self):
    raise NotImplementedError

  def step(self, action):
    raise NotImplementedError

  def preprocess(self, frame):
    raise NotImplementedError
