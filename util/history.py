from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class History(object):

  def __init__(self, size, observed_dims):
    self.history = np.zeros([size] + observed_dims, dtype=np.float32)
    self.length = 0

  def append(self, frame):
    assert (frame.shape == self.history.shape[1:])
    self.history[:-1] = self.history[1:]
    self.history[-1] = frame
    self.length = min(self.length + 1, self.history.shape[0])

  def reset(self):
    self.history *= 0
    self.length = 0

  def filled(self):
    return self.length == self.history.shape[0]

  def get(self):
    return self.history
