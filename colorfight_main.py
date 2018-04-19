from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from colorfight.network import ColorFightNetwork
import colorfight.colorfight_env as cf_env
from colorfight.colorfight_env import ColorFightEnv
from colorfight.agent import ColorFightAgent

import glog as log

flags = tf.app.flags

# Running
flags.DEFINE_boolean('train', True, 'Whether to run the training iteration')
flags.DEFINE_integer('train_steps', 100000, 'Number of training steps')
flags.DEFINE_boolean('play', True, 'Whether to run the playing iteration')

flags.DEFINE_integer(
    'play_steps', 100000,
    'Number of steps after training to play in the environment')
flags.DEFINE_boolean('save_checkpoints', True, 'Whether to save checkpoints')
flags.DEFINE_boolean('save_summaries', True, 'Whether to save summaries')
flags.DEFINE_boolean('restore', True, 'Whether to restore from checkpoints')
flags.DEFINE_string('checkpoint_restore', 'colorfight/checkpoints/',
                    'Path to checkpoint reading directory')

flags.DEFINE_string('checkpoint_store', 'colorfight/checkpoints/',
                    'Path to checkpoint saving directory')
flags.DEFINE_integer(
    'checkpoint_secs', 120,
    'Number of seconds during training to wait between checkpoints')
flags.DEFINE_integer('checkpoint_steps', None,
                     'Number of steps to wait between checkpoints')

flags.DEFINE_string('summaries_dir', 'colorfight/summaries/',
                    'Directory to save summaries')
flags.DEFINE_integer(
    'summary_secs', 60,
    'Number of seconds during training to wait between summaries')
flags.DEFINE_integer('summary_steps', None,
                     'Number of steps to wait between summaries')

flags.DEFINE_string('model_base_name', 'model.ckpt',
                    'Name of file to save the model')
flags.DEFINE_integer('log_step', 100,
                     'Number of steps to wait between log outputs')

#Agent
flags.DEFINE_float('gamma', 0.8, 'Reward discount factor')
flags.DEFINE_float('actor_learning_rate', 0.00001, 'Learning rate of the agent')
flags.DEFINE_float('critic_learning_rate', 0.0001,
                   'Learning rate of the critic')
flags.DEFINE_integer('memory_size', 100, 'Size of the replay memory')
flags.DEFINE_integer('batch_size', 32, 'Batch size per iteration')

#Debug
flags.DEFINE_string('verbosity', 'INFO', 'Logging verbosity')

conf = flags.FLAGS

if __name__ == '__main__':
  log.setLevel(conf.verbosity)
  env = ColorFightEnv('reinforcemnt_v0')

  actor_network = ColorFightNetwork('actor')
  critic_network = ColorFightNetwork('critic')

  actor_network.build_output_op(
      cf_env.GRID_SIZE,
      cf_env.GRID_INPUT_SHAPE,
      cf_env.EXTRA_INPUT_SIZE, [6, 6],
      6,
      32,
      2, [3, 3],
      actor=True,
      trainable=True)

  critic_network.build_output_op(
      cf_env.GRID_SIZE,
      cf_env.GRID_INPUT_SHAPE,
      cf_env.EXTRA_INPUT_SIZE, [6, 6],
      6,
      32,
      2, [3, 3],
      actor=False,
      trainable=True)

  agent = ColorFightAgent(env, actor_network, critic_network,
                          cf_env.ACTION_SPACE, cf_env.GRID_INPUT_SHAPE,
                          cf_env.EXTRA_INPUT_SIZE, conf)

  log.info('Creating computation graph')
  agent.create_main_graph()
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

  log.info('Start training')
  agent.train(sess)
  sess.close()

  if conf.play:
    log.info('New Session')
    log.info('Restoring from %s' % (conf.checkpoint_restore))
    sess = tf.train.MonitoredTrainingSession(
        checkpoint_dir=conf.checkpoint_restore)
    log.info('Start playing')
    agent.play(conf.play_steps, sess)
    sess.close()
