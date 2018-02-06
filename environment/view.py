from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# Interface to represent any view.
# It contains the rendering information, but not the game mechanics
class View(object):

  def __init__(self, name):
    pass

  def render(self):
    pass
