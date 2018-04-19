from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from environment.environment import Environment

import numpy as np

from colorfight import colorfight as cf_API
import glog as log

# Maximum number of players excluding itself
MAX_NUM_PLAYERS = 14

GRID_SIZE = [30, 30]

# Mine (1, 0)
# Empty (1, 0)
# Owner (MAX NUM PLAYERS)
# I'm attacking (1, 0)
# No one is attacking (1, 0)
# Who is attacking (MAX_NUM_PLAYERS)
# Take time (x)
# Gold Cell: (1, 0)
# Energy cell: (1, 0)
# Build type (base or no base) (1, 0)

DEPTH = 1 + 1 + MAX_NUM_PLAYERS + 1 + 1 + MAX_NUM_PLAYERS + 1 + 1 + 1 + 1
GRID_INPUT_SHAPE = [GRID_SIZE[0], GRID_SIZE[1], DEPTH]

# Extra features:
# Energy
# Gold
# CdTime? (1, 0)
EXTRA_INPUT_SIZE = 3

# action space
#   cell loc
#   action_type
#   blast_dir
#   blast_type
ACTION_SPACE = 4


class ColorFightEnv(Environment):

  def __init__(self, ai_name):
    self.ai_name = ai_name
    self.game = cf_API.Game()
    self.grid_state = np.zeros(shape=GRID_INPUT_SHAPE, dtype=np.float32)
    self.extra_state = np.zeros(shape=EXTRA_INPUT_SIZE, dtype=np.float32)
    self.user_dict = None
    self.delta_score = 0
    self.previous_score = 0
    self.score = 0

  def join_game(self):
    if self.game.JoinGame(self.ai_name):
      return True
    else:
      log.warning('Couldn\'t join game')
      return False

  def refresh(self):
    self.game.Refresh()
    self._one_hot_users()
    self.score = self.game.cellNum
    self.delta_score = self.score - self.previous_score
    self.previous_score = self.score

  def step(self, x, y, action, **kwargs):
    if (action == 'attack'):
      _, err, msg = self.game.AttackCell(x, y)
    elif (action == 'build_base'):
      _, err, msg = self.game.BuildBase(x, y)
    elif (action == 'blast'):
      _, err, msg = self.game.Blast(x, y, kwargs['blast_dir'],
                                    kwargs['blast_type'])
    else:
      log.warning('Action is nonsensical')
      return False

    reward = self.delta_score
    if (err is not None):
      reward -= -10
      log.debug("Received error code {}:{}".format(err, msg))
    return reward

  def populate_state(self):
    for x in range(GRID_SIZE[0]):
      for y in range(GRID_SIZE[1]):
        cell = self.game.GetCell(x, y)
        self.grid_state[x, y, 0] = cell.owner == self.game.uid
        self.grid_state[x, y, 1] = cell.owner == 0
        if (cell.owner != self.game.uid and cell.owner != 0):
          self.grid_state[x, y, 2 + self.user_dict[cell.owner]] = 1.0

        self.grid_state[x, y,
                        2 + MAX_NUM_PLAYERS] = cell.attacker == self.game.uid
        self.grid_state[x, y, 3 + MAX_NUM_PLAYERS] = not cell.isTaking
        if (cell.attacker != self.game.uid and cell.attacker != 0):
          self.grid_state[
              x, y, 4 + MAX_NUM_PLAYERS + self.user_dict[cell.attacker]] = 1.0

        self.grid_state[x, y, 4 + 2 * MAX_NUM_PLAYERS] = cell.takeTime
        self.grid_state[x, y, 5 + 2 * MAX_NUM_PLAYERS] = cell.cellType == 'gold'
        self.grid_state[x, y,
                        6 + 2 * MAX_NUM_PLAYERS] = cell.cellType == 'energy'
        self.grid_state[x, y, 7 + 2 * MAX_NUM_PLAYERS] = cell.isBase

    self.extra_state[0] = self.game.energy
    self.extra_state[1] = self.game.gold
    self.extra_state[2] = self.game.cdTime > self.game.currTime

  def _one_hot_users(self):
    users = self.game.users
    indeces = range(len(users))
    self.user_dict = dict(zip([x.id for x in users], indeces))
    self.user_dict.pop(self.game.uid, None)
    if (len(self.user_dict) > MAX_NUM_PLAYERS):
      log.warning('Number of players in game exeeds capable number of players')
