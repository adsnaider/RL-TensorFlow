from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow import logging as log

from agents.policy_gradient import PolicyGradient

from colorfight.memory import Memory


class ColorFightAgent(PolicyGradient):

  def __init__(self, environment, policy_network, value_network, action_space,
               grid_shape, n_extra, conf):
    self.gamma = conf.gamma
    self.actor_learning_rate = conf.actor_learning_rate
    self.critic_learning_rate = conf.critic_learning_rate
    self.train_steps = conf.train_steps
    self.batch_size = conf.batch_size
    self.memory_size = conf.memory_size
    self.log_step = conf.log_step

    self.environment = environment
    self.policy_network = policy_network
    self.value_network = value_network
    self.grid_shape = grid_shape
    self.num_extra_feats = n_extra
    self.history_size = None
    self.history = None
    self.D = Memory(self.memory_size, self.grid_shape, self.num_extra_feats,
                    action_space, self.batch_size)

  def create_main_graph(self):
    with tf.variable_scope('PolicyGradient_main_graph'):
      self.global_step = tf.train.get_or_create_global_step()
      self.global_step_increment_op = tf.assign(self.global_step,
                                                self.global_step + 1)

  def create_training_graph(self):
    with tf.variable_scope('ColorfightPolicyGradient_training_graph'):
      self.post_value = tf.placeholder(
          shape=[None, 1], dtype=tf.float32, name='poststate_value')
      self.reward = tf.placeholder(
          shape=[None, 1], dtype=tf.float32, name='reward')
      self.action_space = tf.placeholder(
          shape=[None, 4], dtype=tf.int32, name='action')

      self.actor_grid_state = self.policy_network.cnn_inputs
      self.actor_extra_state = self.policy_network.fc_inputs
      self.value_grid_state = self.value_network.cnn_inputs
      self.value_extra_state = self.value_network.fc_inputs

      policy_output = self.policy_network.outputs
      value_output = self.value_network.outputs

      td_error = self.reward + self.gamma * self.post_value - value_output
      self.value_loss = tf.reduce_mean(tf.square(td_error), name='value_loss')
      self.value_optimizer = tf.train.GradientDescentOptimizer(
          self.critic_learning_rate).minimize(
              self.value_loss, name='value_optimizer')

      #        batch_range = tf.reshape(
      #  tf.range(tf.shape(self.action_space)[0]), [-1, 1])
      #  print(batch_range)
      #  action_idx = tf.concat(
      #  [batch_range, tf.cast(self.action_space, tf.int32)], axis=1)
      #  loc_idx = tf.concat(
      #  [batch_range, tf.cast(self.action_space[0], tf.int32)], axis=1)
      #  print(action_idx)
      #  policies_active = tf.gather_nd(policy_output, action_idx, 'policy_taken')

      log_prob = tf.log(policy_output, name='log_prob')
      self.actor_optimizer = tf.train.GradientDescentOptimizer(
          self.actor_learning_rate)
      not_loss = -tf.reduce_mean(
          tf.multiply(tf.stop_gradient(td_error), log_prob))
      grads = self.actor_optimizer.compute_gradients(not_loss)
      actor_train_step = self.actor_optimizer.apply_gradients(grads)

      self.train_op = tf.group(actor_train_step, self.value_optimizer,
                               self.global_step_increment_op)

      value_summary = tf.summary.scalar('value_loss', self.value_loss)
      self.summary_op = tf.summary.merge_all()

  def _get_next_action(self, grid_state, extra_state, sess):
    outs = self.policy_network.calc_outputs(grid_state, extra_state, sess)[0]
    grid_width = self.grid_shape[0]
    grid_height = self.grid_shape[1]

    loc_action = np.random.choice(
        range(grid_width * grid_height), p=outs[0:(grid_width * grid_height)])
    loc_action = np.array([loc_action // grid_width, loc_action % grid_width])
    action_type = np.random.choice(
        3, p=outs[(grid_width * grid_height):(grid_width * grid_height) + 3])
    blast_dir = np.random.choice(
        3,
        p=outs[(grid_width * grid_height) + 3:(grid_width * grid_height) + 6])
    blast_type = np.random.choice(
        2,
        p=outs[(grid_width * grid_height) + 6:(grid_width * grid_height) + 8])

    return loc_action, action_type, blast_dir, blast_type, outs

  @staticmethod
  def _convert_action(loc, action_type, blast_dir, blast_type):
    x, y = loc[0], loc[1]
    actions = ['attack', 'build_base', 'blast']
    blast_dirs = ['square', 'vertical', 'horizontal']
    blast_types = ['attack', 'defense']
    action = actions[action_type]
    blast_dir = blast_dirs[blast_dir]
    blast_type = blast_types[blast_type]

    return x, y, action, blast_dir, blast_type

  def _train_step(self, step_context):
    step = step_context.session.run(self.global_step)
    self.environment.populate_state()
    grid_state, extra_state = self.environment.grid_state, self.environment.extra_state

    grid_state = np.expand_dims(grid_state, axis=0)
    extra_state = np.expand_dims(extra_state, axis=0)

    loc, action_num, blast_dir_num, blast_type_num, output = self._get_next_action(
        grid_state, extra_state, step_context.session)
    x, y, action, blast_dir, blast_type = self._convert_action(
        loc, action_num, blast_dir_num, blast_type_num)
    reward = self.environment.step(
        x, y, action, blast_dir=blast_dir, blast_type=blast_type)
    self.D.add(grid_state, extra_state, action_num, reward)

    # yapf: disable
    prestate_grid, prestate_extra, action, reward, poststate_grid, poststate_extra = self.D.sample()
    # yapf: enable

    if prestate_grid is not None and prestate_extra is not None and \
        action is not None and reward is not None and \
        poststate_grid is not None and poststate_extra is not None:

      post_value = np.reshape(
          step_context.session.run(
              self.value_network.outputs,
              feed_dict={
                  self.value_network.cnn_inputs: poststate_grid,
                  self.value_network.fc_inputs: poststate_extra
              }), [-1, 1])
      action = np.reshape(action, [-1, 4])
      reward = np.reshape(reward, [-1, 1])

      _, value_loss = step_context.run_with_hooks(
          [self.train_op, self.value_loss],
          feed_dict={
              self.post_value: post_value,
              self.actor_grid_state: prestate_grid,
              self.actor_extra_state: prestate_extra,
              self.value_grid_state: prestate_grid,
              self.value_extra_state: prestate_extra,
              self.reward: reward,
              self.action_space: action
          })
      if (step % self.log_step == 0):
        log.debug('Step {}/{:<6} Value loss {:<16.6f} Score {:<10}'.format(
            step, self.train_steps, value_loss, self.environment.score))

  def train(self, sess):
    if (self.actor_optimizer is None):
      raise Exception('create_training_graph must be called before train')

    self.D.clear()
    if self.environment.join_game():
      while not sess.should_stop():
        self.environment.refresh()
        sess.run_step_fn(self._train_step)

  def play(self, steps, sess):
    if self.environment.join_game():
      while not sess.should_stop() and steps > 0:
        steps -= 1
        self.environment.refresh()
        grid_state, extra_state = self.environment.grid_state, self.environment.extra_state
        loc, action_type, blast_dir, blast_type, output = self._get_next_action(
            grid_state, extra_state, sess)
        x, y, action, blast_dir, blast_type = self._convert_action(
            loc, action_type, blast_dir, blast_type)
        self.environment.step(
            x, y, action, blast_dir=blast_dir, blast_type=blast_type)
        sess.run(self.global_step_increment_op)
