from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from colorfight.memory import Memory


class DeepQColorFight():

  def __init__(self, environment, q_network, action_space, grid_shape, n_extra,
               logger, conf):
    self.gamma = conf.gamma
    self.learning_rate = conf.actor_learning_rate
    self.train_steps = conf.train_steps
    self.batch_size = conf.batch_size
    self.memory_size = conf.memory_size
    self.log_step = conf.log_step
    self.epsilon = initial_epsilon
    self.final_epsilon = conf.final_epsilon
    self.epsilon_decay = conf.epsilon_decay
    self.plot = conf.plot

    self.environment = environment
    self.q_network = q_network
    self.grid_shape = grid_shape
    self.num_extra_feats = n_extra
    self.loggger = logger
    self.D = Memory(self.memory_size, self.grid_shape, self.num_extra_feats,
                    action_space, self.batch_size)

    plt.ion()

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
      self.current_reward = tf.placeholder(
          shape=[], dtype=tf.float32, name='curr_reward')

      self.q_grid_state = self.q_network.cnn_inputs
      self.q_extra_state = self.q_network.fc_inputs

      q_output = self.q_network.outputs
      value_output = self.value_network.outputs
      self.value_output = value_output

      td_error = self.reward + self.gamma * self.post_value - value_output
      self.value_loss = tf.reduce_mean(tf.square(td_error), name='value_loss')
      self.value_optimizer = tf.train.AdamOptimizer(
          self.critic_learning_rate).minimize(
              self.value_loss, name='value_optimizer')

      loc_start = 0
      action_type_start = loc_start + self.grid_shape[0] * self.grid_shape[1]
      blast_dir_start = action_type_start + 3
      blast_type_start = blast_dir_start + 3

      batch_range = tf.reshape(
          tf.range(tf.shape(self.action_space)[0]), [-1, 1])
      loc_idx = tf.concat(
          [
              batch_range,
              tf.reshape(tf.cast(self.action_space[:, 0], tf.int32), [-1, 1]) +
              loc_start
          ],
          axis=1)
      action_type_idx = tf.concat(
          [
              batch_range,
              tf.reshape(tf.cast(self.action_space[:, 1], tf.int32), [-1, 1]) +
              action_type_start
          ],
          axis=1)
      blast_dir_idx = tf.concat(
          [
              batch_range,
              tf.reshape(tf.cast(self.action_space[:, 2], tf.int32), [-1, 1]) +
              blast_dir_start
          ],
          axis=1)
      blast_type_idx = tf.concat(
          [
              batch_range,
              tf.reshape(tf.cast(self.action_space[:, 3], tf.int32), [-1, 1]) +
              blast_type_start
          ],
          axis=1)

      idx = tf.stack(
          [loc_idx, action_type_idx, blast_dir_idx, blast_type_idx], axis=1)
      active_policy = tf.gather_nd(policy_output, idx, 'policy')
      self.active_policy = active_policy

      log_prob = tf.log(active_policy, name='log_prob')

      self.actor_optimizer = tf.train.AdamOptimizer(self.actor_learning_rate)
      self.not_loss = -tf.reduce_sum(
          tf.multiply(tf.stop_gradient(td_error), log_prob))
      grads = self.actor_optimizer.compute_gradients(self.not_loss)
      actor_train_step = self.actor_optimizer.apply_gradients(grads)

      self.train_op = tf.group(actor_train_step, self.value_optimizer,
                               self.global_step_increment_op)

      value_summary = tf.summary.scalar('value_loss', self.value_loss)
      reward_summary = tf.summary.scalar('rewards', self.current_reward)
      self.summary_op = tf.summary.merge_all()
