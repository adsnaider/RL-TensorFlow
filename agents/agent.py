from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import random

from util.history import History
from util.replay_memory import ReplayMemory

from tensorflow import logging as log


# Interface to represent any agent
class Agent(object):

  def __init__(self, environment, pred_network, conf):
    self.actions = environment.actions
    self.environment = environment
    self.pred_network = pred_network
    self.history_size = conf.history_size
    self.observation_dims = conf.observation_dims
    self.epsilon = conf.epsilon
    self.history = History(self.history_size, self.observation_dims)

  def train(self, sess):
    raise NotImplementedError('Abstract method must be overriden')

  def play(self, iterations, sess):
    self.history.reset()
    obsevation, _, _ = self.environment.new_game()
    for i in xrange(iterations):
      self.history.append(observation)
      # At the beginning action might be wrong since history hasn't been filled yet
      action = self.predict(self.history.get(), sess)
      observation, score, terminal = self.environment.step(action)
      log.debug('Step: {}\tScore: {}\tTerminal: {}'.format(i, score, terminal))

  def predict(self, state, sess):
    return self.pred_network.calc_actions(state, sess)

  def _random_action(self):
    return random.randint(0, self.actions)

  def _get_next_action(self, state, sess):
    r = random.uniform(0, 1)
    if (r < self.epsilon):
      return self._random_action()
    else:
      return self.predict(state[np.newaxis, ...], sess)
