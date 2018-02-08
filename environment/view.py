from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pygame


class Drawable(object):

  def draw(self, screen):
    raise NotImplementedError


# Interface to represent any view.
# It contains the rendering information, but not the game mechanics
class View(object):

  def __init__(self, name, screen_dims, fps):
    self.name = name
    self.screen_dims = screen_dims
    self.screen = None
    self.present_frame = None

  def create_screen(self):
    self.screen = pygame.display.set_mode(self.screen_dims)

  def get_current_frame(self):
    return self.present_frame

  def update(drawable_list, screen_color):
    pygame.event.pump()
    self.screen.fill(screen_color)
    for item in drawable_list:
      item.draw(self.screen)
    self.present_frame = pygame.surfarray.array3d(pygame.display.get_surface())

  def render(self):
    pygame.display.flip()
