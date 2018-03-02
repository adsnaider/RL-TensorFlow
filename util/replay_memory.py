from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class ReplayMemory(object):

  def __init__(self, memory_size, history_size, observation_dims, batch_size):
    self.memory_size = memory_size
    self.history_size = history_size
    self.observation_dims = observation_dims

    self.observations = np.zeros([memory_size] + observation_dims, np.float32)
    self.actions = np.zeros([memory_size], dtype=np.uint8)
    self.rewards = np.zeros([memory_size], dtype=np.uint8)
    self.terminals = np.zeros([memory_size], np.bool)
    self.batch_size = batch_size

    self.prestates = np.zeros(
        [batch_size, history_size] + observation_dims, dtype=np.float32)
    self.poststates = np.zeros(
        [batch_size, history_size] + observation_dims, dtype=np.float32)

    self.count = 0
    self.index = 0

  def add(self, observation, action, reward, terminal):
    self.observations[self.index, ...] = observation
    self.actions[self.index] = action
    self.rewards[self.index] = reward
    self.terminals[self.index] = terminal

    self.count = min(self.count + 1, self.memory_size)
    self.index += 1
    self.index %= self.memory_size

  def filled(self):
    return self.count == self.memory_size

  def sample(self):
    assert (self.filled())
    possible = range(self.memory_size)
    possible[:] = [
        i for i in range(len(possible))
        if not (
            ((i - self.index) % self.memory_size + 1 < self.history_size) or (
                (self.index - i) % self.memory_size < self.history_size) or
            (self.terminals[[
                x % self.memory_size
                for x in range(i - self.history_size, i + self.history_size - 1)
            ]].any()))
    ]

    batch_idx = np.random.choice(possible, self.batch_size, replace=False)

    for i, x in enumerate(batch_idx):
      self.prestates[i, ...], self.poststates[i, ...] = self._retrieve_state(x)

    action = self.actions[batch_idx - 1]
    reward = self.rewards[batch_idx - 1]
    terminal = self.terminals[batch_idx - 1]

    return self.prestates, action, reward, self.poststates, terminal

  def _retrieve_state(self, idx):
    prestate = np.zeros(
        [self.history_size] + self.observation_dims, dtype=np.float32)
    poststate = np.zeros(
        [self.history_size] + self.observation_dims, dtype=np.float32)
    for i in range(self.history_size):
      prestate[i, ...] = self.observations[(
          idx + i - self.history_size) % self.memory_size]
      poststate[i, ...] = self.observations[(idx + i) % self.memory_size]

    return prestate, poststate


if __name__ == '__main__':
  rm = ReplayMemory(20, 3, [1], 5)
  for i in range(28):
    rm.add([i], 0, i, i % 10 == 0)

  print('Observation:\n')
  print(rm.observations)
  print('Samples:\n')
  pre, act, rew, post, term, possible = rm.sample()
  print('Prestates:\n', pre)
  print('Post:\n', post)
  print('Possible\n', possible)
  print('Reward:\n', rew)
  print('Action:\n', act)
  print('Terminal:\n', term)
