from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import pygame


class Drawable(object):

  def draw(self, screen, ratio):
    raise NotImplementedError


class DrawableRect(Drawable):

  def __init__(self, x, y, width, height, color):
    self.x = x
    self.y = y
    self.width = width
    self.height = height
    self.color = color

  def draw(self, screen, ratio):
    obj = pygame.Rect(self.x * ratio[0], self.y * ratio[1],
                      self.width * ratio[0], self.height * ratio[1])
    pygame.draw.rect(screen, self.color, obj)


# Interface to represent any view.
# It contains the rendering information, but not the game mechanics
class View(object):

  def __init__(self, name, screen_dims, window_dims):
    self.name = name
    self.screen_dims = np.array(screen_dims)
    self.window_dims = np.array(window_dims)
    self.ratio = self.window_dims / self.screen_dims

    self.screen = None
    self.present_frame = None

  def create_screen(self):
    self.screen = pygame.display.set_mode(self.window_dims)

  def get_current_frame(self):
    return self.present_frame

  def update(self, drawable_list, screen_color):
    pygame.event.pump()
    self.screen.fill(screen_color)
    for item in drawable_list:
      item.draw(self.screen, self.ratio)
    self.present_frame = pygame.surfarray.array3d(pygame.display.get_surface())

  def render(self):
    pygame.display.flip()
