from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class ReplayMemory(object):

  def __init__(self, memory_size, history_size, observation_dims):
    self.memory_size = memory_size
    self.history_size = history_size
    self.observations_dims = observation_dims

    self.observations = np.zeros([memory_size] + observation_dims, np.float32)
    self.actions = np.zeros([memory_size], dtype=np.uint8)
    self.rewards = np.zeros([memory_size], dtype=np.uint8)
    self.terminals = np.zeros([memory_size], np.bool)

    self.prestates = np.zeros(
        [memory_size, history_size] + observation_dims, dtype=np.float32)
    self.poststates = np.zeros(
        [memory_size, history_size] + observation_dims, dtype=np.float32)

    self.count = 0
    self.index = 0

  def add(self, observation, action, reward, terminal):
    self.observations[self.index, ...] = observation
    self.actions[self.index] = action
    self.rewards[self.index] = reward
    self.terminals[self.index] = terminal

    self.count = max(self.count + 1, self.memory_size)
    self.index += 1
    self.index %= self.memory_size

  def filled():
    return self.count == self.memory_size

  def sample(self, batch_size):
    assert (self.count == self.memory_size)
    possible = range(self.index) + range(self.index + self.history_size - 1,
                                         self.memory_size)
    possible[:] = [
        x for x in possible
        if not self.terminals[(x - self.history_size):x].any()
    ]

    batch_idx = np.random.choice(possible, batch_size, replace=False)

    for i, x in enumerate(batch_idx):
      self.prestates[i, ...], self.poststates[i, ...] = _retrieve_state(x)

    action = self.actions[batch_idx]
    reward = self.rewards[batch_idx]
    terminal = self.terminals[batch_idx]

    return self.prestates, action, reward, self.poststates, terminal

  def _retrieve_state(self, idx):
    pass
    return None, None
