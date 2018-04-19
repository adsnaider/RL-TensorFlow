from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class Memory(object):

  def __init__(self, memory_size, grid_dims, n_extra, action_space, batch_size):
    self.memory_size = memory_size
    self.grid_dims = grid_dims
    self.n_extra = n_extra
    self.batch_size = batch_size

    self.grid_observations = np.zeros([memory_size] + grid_dims, np.float32)
    self.extra_observations = np.zeros([memory_size, n_extra], np.float32)
    self.actions = np.zeros([memory_size, action_space], np.uint8)
    self.rewards = np.zeros([memory_size], np.uint8)

    self.batch_size = batch_size
    self.prestates_grid = np.zeros([batch_size] + grid_dims, np.float32)
    self.prestates_extra = np.zeros([batch_size, n_extra], np.float32)
    self.poststates_grid = np.zeros([batch_size] + grid_dims, np.float32)
    self.poststates_extra = np.zeros([batch_size, n_extra], np.float32)

    self.index = 0
    self.count = 0

  def add(self, grid_state, extra_state, action, reward):
    self.grid_observations[self.index, ...] = grid_state
    self.extra_observations[self.index, ...] = extra_state
    self.actions[self.index, ...] = action
    self.rewards[self.index] = reward

    self.index = (self.index + 1) % self.memory_size
    self.count = min(self.count + 1, self.memory_size)

  def filled(self):
    return self.count == self.memory_size

  def clear(self):
    self.grid_observations *= 0
    self.extra_observations *= 0
    self.actions *= 0
    self.rewards *= 0
    self.count = 0

    self.prestates_grid *= 0
    self.poststates_grid *= 0
    self.prestates_extra *= 0
    self.poststates_extra *= 0

  def sample(self):
    possible = np.arange(self.count)
    if (self.count <= self.batch_size):
      batch_idx = possible
    else:
      batch_idx = np.random.choice(possible, self.batch_size, replace=False)
    self.prestates_grid = self.grid_observations[batch_idx]
    self.poststates_grid = self.grid_observations[(batch_idx + 1) % self.count]
    self.prestates_extra = self.extra_observations[batch_idx]
    self.poststates_extra = self.extra_observations[(
        batch_idx + 1) % self.count]

    action = self.actions[batch_idx]
    reward = self.rewards[batch_idx]

    return self.prestates_grid, self.prestates_extra, action, reward, self.poststates_grid, self.poststates_extra
