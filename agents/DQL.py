from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow import logging as log

import sys

from util.history import History
from util.replay_memory import ReplayMemory

from agents.agent import Agent


class DQL(Agent):

  def __init__(self, environment, pred_network, target_network, conf):
    super(DQL, self).__init__(environment, pred_network, conf)
    self.target_network = target_network
    self.target_network.create_copy_op(pred_network)

    self.gamma = conf.gamma
    self.learning_rate = conf.learning_rate
    self.train_steps = conf.train_steps
    self.observation_time = conf.observation_time
    self.memory_size = conf.memory_size
    self.batch_size = conf.batch_size
    self.step_size = conf.step_size

    self.D = ReplayMemory(self.memory_size, self.history_size,
                          self.observation_dims, self.batch_size)

    self.reward_sum = 0
    self.optimizer = None

  def create_main_graph(self):
    with tf.variable_scope('DQL_main_graph'):
      self.global_step = tf.train.get_or_create_global_step()

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
      self.target_output = self.target_network.outputs

      batch_range = tf.range(tf.shape(self.action_input)[0])
      action_idx = tf.stack(
          [batch_range, tf.cast(self.action_input, tf.int32)], axis=1)
      pred_value = tf.gather_nd(pred_output, action_idx, 'prediction_value')

      target_action = self.target_network.actions
      action_idx = tf.stack(
          [batch_range, tf.cast(target_action, tf.int32)], axis=1)
      target_value = tf.gather_nd(
          self.target_output, action_idx, name='target_value')

      # If terminal is True (aka 1), then gt is the rewards.
      # If terminal is False (aka 0), then gt is the discounted reward.
      gt = tf.multiply(
          tf.constant(1, dtype=tf.float32) - self.terminal_input,
          self.reward_input + self.gamma * target_value) + tf.multiply(
              self.terminal_input, self.reward_input)
      self.loss = tf.losses.mean_squared_error(gt, pred_value)

      self.optimizer = tf.train.GradientDescentOptimizer(
          self.learning_rate).minimize(self.loss, self.global_step)

      self.max_q = tf.reduce_max(pred_value, name='max_q')
      loss_summary = tf.summary.scalar('loss', self.loss)
      q_summary = tf.summary.scalar('max_q', self.max_q)
      self.summary_op = tf.summary.merge_all()

  def _train_step(self, step_context):
    step = step_context.session.run(self.global_step)
    self.history.append(self.observation)
    action = self._get_next_action(self.history.get(), step,
                                   step_context.session)
    q_value = step_context.session.run(
        self.max_q,
        feed_dict={
            self.prestate_input: self.history.get()[np.newaxis, ...],
            self.action_input: action
        })
    self.observation, reward, terminal = self.environment.step(action[0])
    self.reward_sum += reward
    self.D.add(self.observation, action[0], reward, terminal)
    if (self.D.filled()):
      if step % self.observation_time == 0:
        log.debug('Running copy operation')
        self.target_network.run_copy_op(step_context.session)
      prestates, actions, rewards, poststates, terminal = self.D.sample()
      _, loss_eval = step_context.run_with_hooks(
          [self.optimizer, self.loss],
          feed_dict={
              self.prestate_input: prestates,
              self.poststate_input: poststates,
              self.reward_input: rewards,
              self.action_input: actions,
              self.terminal_input: terminal.astype(np.int32)
          })
      log.debug(
          'Step {}/{:<6} Loss {:<16.6f} Q {:<16.6f} Rewards Received {:<10} Epsilon {:<10.3f}'.
          format(step, self.train_steps, loss_eval, q_value, self.reward_sum,
                 self.epsilon))

  def train(self, sess):
    if (self.optimizer is None):
      raise Exception('create_trianing_graph must be called before train.')

    self.history.reset()
    self.observation, _, _ = self.environment.new_game()
    while not sess.should_stop():
      sess.run_step_fn(self._train_step)
