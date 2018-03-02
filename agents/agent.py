from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import random
import time

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
    self.initial_epsilon = conf.initial_epsilon
    self.epsilon_decay_rate = conf.epsilon_decay_rate
    self.epsilon_decay_step = conf.epsilon_decay_step
    self.epsilon = self.initial_epsilon
    self.history = History(self.history_size, self.observation_dims)

  def train(self, sess):
    raise NotImplementedError('Abstract method must be overriden')

  def play(self, iterations, FPS, sess):
    self.history.reset()
    observation, _, _ = self.environment.new_game()
    for i in xrange(iterations):
      self.history.append(observation)
      # At the beginning action might be wrong since history hasn't been filled yet
      action = self.predict(self.history.get(), sess)
      observation, _, terminal = self.environment.step(action)
      score = self.environment.score
      log.debug('Step: {:<6} Score: {:<6} Terminal: {}'.format(
          i, score, terminal))
      #  time.sleep(1 / FPS)

  def predict(self, state, sess):
    return self.pred_network.calc_actions(state[np.newaxis, ...], sess)

  def _random_action(self):
    return random.randint(0, self.actions)

  def _get_next_action(self, state, step, sess):
    self.epsilon = self.initial_epsilon * pow(self.epsilon_decay_rate,
                                              step // self.epsilon_decay_step)
    r = random.uniform(0, 1)
    if (r < self.epsilon):
      return np.array([self._random_action()])
    else:
      return self.predict(state, sess)
