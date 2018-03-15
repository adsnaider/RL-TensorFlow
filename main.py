from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from shutil import rmtree

import numpy as np

import tensorflow as tf
from tensorflow import logging as log
from tensorflow.python import debug as tf_debug

from environment.pong import Pong

from agents.DQL import DQL
from agents.policy_gradient import PolicyGradient

from networks.CNN import CNN

flags = tf.app.flags

# Network params
flags.DEFINE_string('network_type', 'CNN', 'Type of network to use')
flags.DEFINE_integer('depth', 8, 'CNN output depth')
flags.DEFINE_integer('hidden_size', 32, 'Size of the hidden layer for CNN')
flags.DEFINE_integer('num_conv', 3,
                     'Number of convolutional layers for the CNN')
flags.DEFINE_string('kernel_dims', '[6, 6]',
                    'The dimensions of the kernel for the CNN')
flags.DEFINE_string('strides', '[3, 3]',
                    'The dimmensions of the strides for the CNN')
flags.DEFINE_string('hidden_activation_fn', 'tf.nn.relu',
                    'The activation function for the hidden layer')
flags.DEFINE_string('output_activation_fn', 'None',
                    'The activation function for the output layer')

# Agent params
flags.DEFINE_string('agent', '', 'Type of agent to use')
flags.DEFINE_float('gamma', 0.995, 'Reward discount factor')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate of the agent')
flags.DEFINE_float('actor_learning_rate', 0.1, 'Learning rate of the agent')
flags.DEFINE_float('critic_learning_rate', 0.1, 'Learning rate of the agent')
flags.DEFINE_integer('history_size', 2, 'Number of observations per state')
flags.DEFINE_integer('observation_time', 1500,
                     'Observation time before running network copy operation')
flags.DEFINE_integer('memory_size', 1000, 'Size of the replay memory')
flags.DEFINE_string('observation_dims', '[80, 80, 1]',
                    'The dimensions observed by the agent')
flags.DEFINE_float('initial_epsilon', 1.00,
                   'Initial probability of picking a random action')
flags.DEFINE_float('epsilon_decay_rate', 0.97, 'The decay rate for epsilon')
flags.DEFINE_integer('epsilon_decay_step', 800, 'The decay step for epsilon')
flags.DEFINE_integer('batch_size', 32, 'Batch size per iteration')
flags.DEFINE_integer('step_size', 5, 'Number of steps before observing again')

# Running
flags.DEFINE_integer('seed', 0,
                     'Seed to feed the numpy random number generator')
flags.DEFINE_boolean('train', True, 'Whether to run the training iteration')
flags.DEFINE_integer('train_steps', 100000, 'Number of training steps')
flags.DEFINE_boolean('play', True, 'Whether to run the playing iteration')
flags.DEFINE_integer('fps', 45, 'Speed to run the playing iterations')

flags.DEFINE_integer(
    'play_steps', 10000,
    'Number of steps after training to play in the environment')
flags.DEFINE_boolean('save_checkpoints', True, 'Whether to save checkpoints')
flags.DEFINE_boolean('save_summaries', True, 'Whether to save summaries')
flags.DEFINE_boolean('restore', True, 'Whether to restore from checkpoints')
flags.DEFINE_string('checkpoint_restore', 'checkpoints/',
                    'Path to checkpoint reading directory')

flags.DEFINE_string('checkpoint_store', 'checkpoints/',
                    'Path to checkpoint saving directory')
flags.DEFINE_integer(
    'checkpoint_secs', 120,
    'Number of seconds during training to wait between checkpoints')
flags.DEFINE_integer('checkpoint_steps', None,
                     'Number of steps to wait between checkpoints')

flags.DEFINE_string('summaries_dir', 'summaries/',
                    'Directory to save summaries')
flags.DEFINE_integer(
    'summary_secs', None,
    'Number of seconds during training to wait between summaries')
flags.DEFINE_integer('summary_steps', 20,
                     'Number of steps to wait between summaries')

flags.DEFINE_string('model_base_name', 'model.ckpt',
                    'Name of file to save the model')

# Environment
flags.DEFINE_string('environment', 'pong', 'What environment to run')
flags.DEFINE_string('window_name', '', 'Name of rendering window')
flags.DEFINE_string('screen_dims', '[400, 400]', 'Rendered window dimensions')
flags.DEFINE_boolean('rendering', True, 'Whether to render the environment')

#Debug
flags.DEFINE_boolean('debug', False,
                     'Whether to start the training session in debugging mode')
flags.DEFINE_boolean(
    'override', False,
    'Whether to delete the current checkpoints and summaries before starting')

# Logger
flags.DEFINE_string('verbosity', 'INFO', 'Logger level')

conf = flags.FLAGS

conf.kernel_dims = eval(conf.kernel_dims)
conf.strides = eval(conf.strides)
conf.observation_dims = eval(conf.observation_dims)
conf.screen_dims = eval(conf.screen_dims)
conf.hidden_activation_fn = eval(conf.hidden_activation_fn)
conf.output_activation_fn = eval(conf.output_activation_fn)

log.set_verbosity(conf.verbosity)

np.random.seed(conf.seed)
tf.set_random_seed(conf.seed)

if __name__ == '__main__':
  if (conf.override):
    log.info('Deleting checkpoints and summaries')
    rmtree(conf.checkpoint_dir, ignore_errors=True)
    rmtree(conf.summaries_dir, ignore_errors=True)
  log.info('Creating environment %s' % (conf.environment))
  if conf.environment == 'pong':
    environment = Pong(conf.window_name, conf.observation_dims,
                       conf.screen_dims, conf.rendering)
  else:
    raise ValueError('environment not defined')

  log.info('Creating agent %s' % (conf.agent))
  if conf.agent == 'DQL':
    log.info('Creating network %s' % (conf.network_type))
    if conf.network_type == 'CNN':
      pred = CNN('predict_CNN')
      target = CNN('target_CNN')
      pred.build_output_op(
          conf.depth,
          conf.hidden_size,
          environment.actions,
          conf.num_conv,
          conf.kernel_dims,
          conf.strides,
          conf.observation_dims,
          conf.history_size,
          conf.hidden_activation_fn,
          conf.output_activation_fn,
          trainable=True)

      target.build_output_op(
          conf.depth,
          conf.hidden_size,
          environment.actions,
          conf.num_conv,
          conf.kernel_dims,
          conf.strides,
          conf.observation_dims,
          conf.history_size,
          conf.hidden_activation_fn,
          conf.output_activation_fn,
          trainable=False)

    else:
      raise ValueError('network not defined')
    agent = DQL(environment, pred, target, conf)
  elif conf.agent == 'PolicyGradient':
    log.info('Creating network %s' % (conf.network_type))
    if conf.network_type == 'CNN':
      actor = CNN('policy_CNN')
      critic = CNN('value_CNN')
      actor.build_output_op(
          conf.depth,
          conf.hidden_size,
          environment.actions,
          conf.num_conv,
          conf.kernel_dims,
          conf.strides,
          conf.observation_dims,
          conf.history_size,
          conf.hidden_activation_fn,
          tf.nn.softmax,
          trainable=True)

      critic.build_output_op(
          conf.depth,
          conf.hidden_size,
          1,
          conf.num_conv,
          conf.kernel_dims,
          conf.strides,
          conf.observation_dims,
          conf.history_size,
          conf.hidden_activation_fn,
          conf.output_activation_fn,
          trainable=True)
    else:
      raise ValueError('network not defined')
    agent = PolicyGradient(environment, actor, critic, conf)
  else:
    raise ValueError('agent not defined')
  agent.create_main_graph()
  if conf.train:
    log.info('Creating computation graph')
    agent.create_training_graph()
    hooks = []
    if conf.save_checkpoints:
      hooks.append(
          tf.train.CheckpointSaverHook(
              checkpoint_dir=conf.checkpoint_store,
              save_secs=conf.checkpoint_secs,
              save_steps=conf.checkpoint_steps,
              saver=tf.train.Saver(),
              checkpoint_basename=conf.model_base_name,
              scaffold=None))

    if conf.save_summaries:
      hooks.append(
          tf.train.SummarySaverHook(
              save_secs=conf.summary_secs,
              save_steps=conf.summary_steps,
              output_dir=conf.summaries_dir,
              summary_writer=None,
              scaffold=None,
              summary_op=agent.summary_op))
    hooks.append(tf.train.StopAtStepHook(last_step=conf.train_steps))

    log.info('New Session')
    if conf.restore:
      log.info('Restoring from %s' % (conf.checkpoint_restore))
      sess = tf.train.MonitoredTrainingSession(
          checkpoint_dir=conf.checkpoint_restore, hooks=hooks)
    else:
      sess = tf.train.MonitoredTrainingSession(hooks=hooks)

    if (conf.debug):
      log.info('Starting session with tfdbg')
      sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    log.info('Start training')
    agent.train(sess)
    sess.close()
  if conf.play:
    log.info('New Session')
    log.info('Restoring from %s' % (conf.checkpoint_restore))
    sess = tf.train.MonitoredTrainingSession(
        checkpoint_dir=conf.checkpoint_restore)
    log.info('Start playing')
    agent.play(conf.play_steps, conf.fps, sess)
    sess.close()
