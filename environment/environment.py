from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# Interface to represent any environment.
# It contains the the game mechanics
class Environment(object):

  # name: The environment name
  # actions: The number of finite actions that the agent can take
  # observation_dims: the size of the screen that will be fed into the network
  # view: the object that will render the screen
  def __init__(self, name, actions, observation_dims, view):
    pass

  def new_game(self):
    pass

  def step(self):
    pass

  def preprocess(self):
    pass
