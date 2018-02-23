from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from util.history import History
from util.replay_memory import ReplayMemory

from agents.agent import Agent

import tensorflow as tf
from tensorflow import logging as log


class DQL(Agent):

  def __init__(self, environment, pred_network, target_network, conf):
    super(DQL, self).__init__(environment, pred_network, conf)
    self.target_network = target_network
    self.target_network.create_copy_op(pred_network)

    self.gamma = conf.gamma
    self.learning_rate = conf.learning_rate
    self.num_steps = conf.num_steps
    self.observation_time = conf.observation_time
    self.memory_size = conf.memory_size
    self.batch_size = conf.batch_size

    self.D = ReplayMemory(self.memory_size, self.history_size,
                          self.observation_dims, self.batch_size)

    self.optimizer = None

  def create_training_graph(self):

    with tf.variable_scope('DQL_train_graph'):
      self.prestate_input = self.pred_network.inputs
      self.poststate_input = self.target_network.inputs

      self.action_input = tf.placeholder(
          shape=[None], dtype=tf.int32, name='actions')
      self.reward_input = tf.placeholder(
          shape=[None], dtype=tf.float32, name='rewards')

      self.terminal_input = tf.placeholder(
          shape=[None], dtype=tf.float32, name='terminal')

      pred_output = self.pred_network.outputs
      pred_value = tf.nn.embedding_lookup(
          pred_output, self.action_input, name='prediction_value')

      self.target_output = self.target_network.outputs
      target_action = self.target_network.actions
      target_value = tf.nn.embedding_lookup(
          self.target_output, target_action, name='target_prediction')

      global_step = tf.train.get_or_create_global_step()

      # If terminal is True (aka 1), then gt is the rewards.
      # If terminal is False (aka 0), then gt is the discounted reward.
      gt = (tf.constant(1, dtype=tf.float32) - self.terminal_input) * (
          self.reward_input + self.gamma * target_value
      ) + self.terminal_input * self.reward_input
      self.loss = tf.losses.mean_squared_error(gt, pred_value)

      self.optimizer = tf.train.GradientDescentOptimizer(
          self.learning_rate).minimize(self.loss, global_step)

      loss_summary = tf.summary.scalar('loss', self.loss)
      q_summary = tf.summary.scalar('q', self.target_output)
      self.summary_op = tf.summary.merge_all()

  def train(self, sess):
    if (self.optimizer is None):
      raise Exception('create_trianing_graph must be called before train.')
    self.history.reset()
    observation, _, _ = self.environment.new_game()
    for step in xrange(self.num_steps):
      if step % self.observation_time == 0:
        self.target_network.run_copy_op(sess)
      self.history.append(observation)
      action = self._get_next_action(self.history.get(), sess)
      observation, score, terminal = self.environment.step(action)
      self.D.add(observation, action, score, terminal)
      if (self.D.filled()):
        prestates, actions, rewards, poststates, terminal = self.D.sample()
        _, loss_eval, q_eval = sess.run(
            [self.optimizer, self.loss, self.target_output],
            feed_dict={
                self.prestate_input: prestates,
                self.poststate_input: poststates,
                self.reward_input: rewards,
                self.action_input: actions,
                self.terminal_input: int(terminal)
            })
        log.debug('Step {}/{}\tLoss {}\tQ {}'.format(step, self.num_steps,
                                                     loss_eval, q_eval))
