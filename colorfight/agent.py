from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from agents.policy_gradient import PolicyGradient

from colorfight.memory import Memory


class ColorFightAgent(PolicyGradient):

  def __init__(self, environment, policy_network, value_network, action_space,
               grid_shape, n_extra, log, conf):
    self.gamma = conf.gamma
    self.actor_learning_rate = conf.actor_learning_rate
    self.critic_learning_rate = conf.critic_learning_rate
    self.train_steps = conf.train_steps
    self.batch_size = conf.batch_size
    self.memory_size = conf.memory_size
    self.log_step = conf.log_step
    self.plot = conf.plot
    self.epsilon = conf.initial_epsilon
    self.epsilon_decay_rate = conf.epsilon_decay_rate
    self.epsilon_decay_step = conf.epsilon_decay_step
    self.final_epsilon = conf.final_epsilon

    self.environment = environment
    self.policy_network = policy_network
    self.value_network = value_network
    self.grid_shape = grid_shape
    self.num_extra_feats = n_extra
    self.history_size = None
    self.history = None
    self.log = log
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

      self.actor_grid_state = self.policy_network.cnn_inputs
      self.actor_extra_state = self.policy_network.fc_inputs
      self.value_grid_state = self.value_network.cnn_inputs
      self.value_extra_state = self.value_network.fc_inputs

      policy_output = self.policy_network.outputs
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

  def _get_next_action(self, grid_state, extra_state, sess):
    grid_width = self.grid_shape[0]
    grid_height = self.grid_shape[1]
    outs = self.policy_network.calc_outputs(grid_state, extra_state, sess)[0]
    if np.random.rand() < self.epsilon:
      loc_action = np.random.choice(grid_width * grid_height)
      action_type = np.random.choice(3)
      blast_dir = np.random.choice(3)
      blast_type = np.random.choice(2)

    else:

      loc_action = np.random.choice(
          grid_width * grid_height, p=outs[:(grid_width * grid_height)])
      action_type = np.random.choice(
          3, p=outs[(grid_width * grid_height):(grid_width * grid_height) + 3])
      blast_dir = np.random.choice(
          3,
          p=outs[(grid_width * grid_height) + 3:(grid_width * grid_height) + 6])
      blast_type = np.random.choice(
          2,
          p=outs[(grid_width * grid_height) + 6:(grid_width * grid_height) + 8])

    return loc_action, action_type, blast_dir, blast_type, outs

  def _convert_action(self, loc, action_type, blast_dir, blast_type):
    x, y = loc % self.grid_shape[0], loc // self.grid_shape[0]
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
    curr_reward = self.environment.step(
        x, y, action, blast_dir=blast_dir, blast_type=blast_type)
    self.D.add(grid_state, extra_state,
               np.array([loc, action_num, blast_dir_num, blast_type_num]),
               curr_reward)

    # yapf: disable
    prestate_grid, prestate_extra, action, reward, poststate_grid, poststate_extra = self.D.sample()
    # yapf: enable

    # Decay epsilon
    if ((step + 1) % self.epsilon_decay_step == 0):
      self.epsilon = self.epsilon_decay_rate * self.epsilon

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

      _, value_loss, not_loss, active_policy = step_context.run_with_hooks(
          [self.train_op, self.value_loss, self.not_loss, self.active_policy],
          feed_dict={
              self.post_value: post_value,
              self.actor_grid_state: prestate_grid,
              self.actor_extra_state: prestate_extra,
              self.value_grid_state: prestate_grid,
              self.value_extra_state: prestate_extra,
              self.reward: reward,
              self.current_reward: curr_reward,
              self.action_space: action
          })
      if (step % self.log_step == 0):
        current_value = step_context.session.run(
            self.value_output,
            feed_dict={
                self.value_network.cnn_inputs: grid_state,
                self.value_network.fc_inputs: extra_state
            })
        self.log.info(
            'Step {}/{:<6} Value loss {:<16.6f} Actor Loss {:<16.6f} Score {:<10} Loc ({}, {}) Value {} Epsilon {}'.
            format(step, self.train_steps, value_loss, not_loss**2,
                   self.environment.score, x, y, current_value[0, 0],
                   self.epsilon))

        action_init = self.grid_shape[0] * self.grid_shape[1]
        blast_dir_init = action_init + 3
        blast_type_init = blast_dir_init + 3

        self.log.info('Action: {}  Blast Dir: {}  Blast Type: {}'.format(
            output[action_init:blast_dir_init],
            output[blast_dir_init:blast_type_init], output[blast_type_init:]))
        if (self.plot):
          plt.imshow(
              np.reshape(output[:(self.grid_shape[0] * self.grid_shape[1])],
                         self.grid_shape[:2]),
              cmap='gray')
          plt.draw()
          plt.pause(0.0001)

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
