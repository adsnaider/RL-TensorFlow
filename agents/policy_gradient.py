from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow import logging as log

from util.history import History
from util.full_history import FullHistory
from util.replay_memory import ReplayMemory

from agents.agent import Agent


class PolicyGradient(Agent):

  def __init__(self, environment, policy_network, value_network, conf):
    super(PolicyGradient, self).__init__(environment, None, conf)
    self.policy_network = policy_network
    self.value_network = value_network

    self.gamma = conf.gamma
    self.actor_learning_rate = conf.actor_learning_rate
    self.critic_learning_rate = conf.critic_learning_rate
    self.train_steps = conf.train_steps
    self.batch_size = conf.batch_size
    self.memory_size = conf.memory_size

    self.state_shape = [self.history_size] + self.observation_dims

    self.D = ReplayMemory(self.memory_size, self.history_size,
                          self.observation_dims, self.batch_size)

    self.actor_optimizer = None
    self.value_optimizer = None

  def create_main_graph(self):
    with tf.variable_scope('PolicyGradient_main_graph'):
      self.global_step = tf.train.get_or_create_global_step()
      self.global_step_increment_op = tf.assign(self.global_step,
                                                self.global_step + 1)

  def create_training_graph(self):
    with tf.variable_scope('PolicyGradient_training_graph'):
      self.post_value = tf.placeholder(
          shape=[None, 1], dtype=tf.float32, name='poststate_value')
      self.reward = tf.placeholder(
          shape=[None, 1], dtype=tf.float32, name='reward')
      self.action = tf.placeholder(
          shape=[None, 1], dtype=tf.int32, name='action')

      self.actor_state = self.policy_network.inputs
      self.value_state = self.value_network.inputs

      policy_output = self.policy_network.outputs
      value_output = self.value_network.outputs

      td_error = self.reward + self.gamma * self.post_value - value_output
      self.value_loss = tf.reduce_mean(tf.square(td_error), name='value_loss')
      self.value_optimizer = tf.train.GradientDescentOptimizer(
          self.critic_learning_rate).minimize(
              self.value_loss, name='value_optimizer')

      batch_range = tf.range(tf.shape(self.action)[0])
      action_idx = tf.stack(
          [batch_range, tf.cast(self.action, tf.int32)[:, 0]], axis=1)
      policy_active = tf.gather_nd(policy_output, action_idx, 'policy_taken')

      log_prob = tf.log(policy_active, name='log_prob')
      self.actor_loss = tf.identity(
          -tf.reduce_mean(
              tf.multiply(tf.stop_gradient(td_error[:, 0]), log_prob)),
          name='actor_loss')
      self.actor_optimizer = tf.train.GradientDescentOptimizer(
          self.actor_learning_rate).minimize(
              self.actor_loss, name='actor_optimizer')
      self.train_op = tf.group(self.actor_optimizer, self.value_optimizer,
                               self.global_step_increment_op)

      value_summary = tf.summary.scalar('value_loss', self.value_loss)
      actor_summary = tf.summary.scalar('actor_loss', self.actor_loss)
      self.summary_op = tf.summary.merge_all()

  def _get_next_action(self, state, sess):
    policy_output = self.policy_network.calc_outputs(state, sess)[0]
    action = np.random.choice(
        range(self.environment.actions), 1, p=policy_output)
    return action, policy_output

  def _train_step(self, step_context):
    step = step_context.session.run(self.global_step)
    self.history.append(self.observation)
    state = np.reshape(self.history.get(), [-1] + self.state_shape)
    action, output = self._get_next_action(state, step_context.session)
    self.observation, reward, terminal = self.environment.step(action)
    self.D.add(self.observation, action, reward, terminal)
    if (terminal):
      self.history.reset()

    prestate, action, reward, poststate, terminal = self.D.sample()

    if prestate is not None and action is not None and reward is not None and poststate is not None and terminal is not None:
      prestate = np.reshape(prestate, [-1] + self.state_shape)
      poststate = np.reshape(poststate, [-1] + self.state_shape)
      action = np.reshape(action, [-1, 1])
      reward = np.reshape(reward, [-1, 1])
      terminal = np.reshape(terminal, [-1, 1])

      post_value = np.reshape(
          step_context.session.run(
              self.value_network.outputs,
              feed_dict={
                  self.value_network.inputs: poststate
              }), [-1, 1])

      post_value = np.multiply(post_value, 1 - terminal.astype(np.int32))
      _, actor_loss, value_loss = step_context.run_with_hooks(
          [self.train_op, self.actor_loss, self.value_loss],
          feed_dict={
              self.post_value: post_value,
              self.actor_state: prestate,
              self.value_state: prestate,
              self.reward: reward,
              self.action: action
          })
      log.debug(
          'Step {}/{:<6} Actor loss {:<16.6f} Value loss {:<16.6f} Score {:<10} Policy Output: {}'.
          format(step, self.train_steps, actor_loss, value_loss,
                 self.environment.score, output))

  def train(self, sess):
    if (self.actor_optimizer is None):
      raise Exception('create_training_graph must be called before train')

    self.reward_sum = 0
    self.history.reset()
    self.observation, _, _ = self.environment.new_game()
    while not sess.should_stop():
      sess.run_step_fn(self._train_step)

  def play(self, iterations, FPS, sess):
    self.history.reset()
    observation, _, _ = self.environment.new_game()
    for i in xrange(iterations):
      self.history.append(observation)
      action = self._get_next_action(self.history.get(), sess)
      observation, _, terminal = self.environment.step(action)
      if (terminal):
        self.history.reset()
      score = self.environment.score
      log.debug('Step: {:<6} Score: {:<6} Terminal: {}'.format(
          i, score, terminal))
