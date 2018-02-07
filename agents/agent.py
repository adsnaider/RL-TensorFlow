from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from .history import History
from .replay_memory import ReplayMemory

import tf.logging as log


# Interface to represent any agent
class Agent(object):

  def __init__(self, environment, pred_network, conf):
    self.environment = environment
    self.pred_network = pred_network
    self.target_network = target_network
    self.conf = conf

    self.D = ReplayMemory(conf.replay_size, conf.history_size,
                          conf.observation_dims)
    self.history = History(conf.history_size, conf.observation_dims)

  def train(self):
    raise NotImplementedError('Abstract method must be overriden')

  def play(self, iterations):
    self.history.reset()
    obsevation, _, _ = self.environment.new_game()
    for i in xrange(iterations):
      self.history.append(observation)
      # At the beginning action might be wrong since history hasn't been filled yet
      action = self.predict(self.history.get())
      observation, score, terminal = self.environment.step(action)
      log.debug('Step: {}\tScore: {}\tTerminal: {}'.format(i, score, terminal))

  def predict(self, state):
    return self.pred_network.calc_actions(state)
