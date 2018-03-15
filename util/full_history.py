from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class FullHistory(object):

  def __init__(self, size, observed_dims):
    self.prestate = np.zeros([size] + observed_dims, dtype=np.float32)
    self.poststate = np.zeros([size] + observed_dims, dtype=np.float32)
    self.actions = np.zeros([size], dtype=np.int32)
    self.rewards = np.zeros([size], dtype=np.float32)
    self.terminals = np.zeros([size], dtype=np.bool)

    self.index = -1
    self.size = size
    self.observed_dims = observed_dims

  def reset(self):
    self.prestate *= 0
    self.poststate *= 0
    self.actions *= 0
    self.rewards *= 0
    self.terminals.fill(False)

  def get(self):
    return self.prestate, self.actions[
        self.index + 1 - self.size], self.rewards[
            self.index + 1 - self.size], self.poststate, self.terminals[
                self.index + 1 - self.size]

  def append(self, observation, action, reward, terminal):
    self.index = (self.index + 1) % self.size
    self.prestate[self.index] = self.poststate[self.index]
    self.poststate[self.index] = observation
    self.actions[self.index] = self.index
    self.rewards[self.index] = reward
    self.terminals[self.index] = terminal
