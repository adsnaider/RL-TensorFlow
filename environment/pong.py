from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pygame
import random
import time

from .environment import Environment
from .view import View, Drawable

import tf.logging as log

import numpy as np
import cv2

# RGB Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)  # Initialize our screen


class DrawableRect(Drawable):

  def __init__(self, x, y, width, height, color):
    self.x = x
    self.y = y
    self.width = width
    self.height = height
    self.color = color

  def draw(self, screen):
    obj = pygame.Rect(self.x, self.y, self.width, self.height)
    pygame.draw.rect(screen, self.color, obj)


class Pong(Environment):

  SCREEN_COLOR = BLACK

  def __init__(self,
               name,
               observation_dims,
               screen_dims=(400, 400),
               rendering=True):
    self.actions = 3  # Up, Down, Nothing
    self.paddle_nn = self.Paddle('LEFT')  # DQN
    self.paddle_ai = self.Paddle('RIGHT')  # simple AI
    self.ball = self.Ball()
    draw_list = [self.paddle_nn, self.paddle_ai, self.ball]
    self.score = 0
    super(Pong, self).__init__(self, name, self.actions, observation_dims,
                               rendering, View(name, screen_dims))

  def reset(self):
    self.paddle_nn.reset(self.screen_dims)
    self.paddle_ai.reset(self.screen_dims)
    self.ball.reset(screen_dims)
    self.score = 0

  def new_game(self):
    self.reset()
    self._update()
    return self.preproces(self.view.get_current_frame()), self.score, False

  def step(self, action):
    # Update NN
    self.paddle_nn.update(action)
    # Update AI
    action = 0
    if (self.paddle_ai.y + self.Paddle.HEIGHT / 2 <
        self.ball.y + self.Ball.HEIGHT / 2):
      action = 1
    elif (self.paddle_ai.y + self.Paddle.HEIGHT / 2 >
          self.ball.y + self.Ball.HEIGHT / 2):
      action = -1
    self.paddle_ai.update(action)

    self.score += self.ball.update(self.paddle_nn, self.paddle_ai,
                                   self.screen_dims)
    self._update()
    return self.preprocess(self.view.get_current_frame()), self.score, False

  def _update(self):
    self.view.update(draw_list, Pong.SCREEN_COLOR)
    if (rendering):
      self.view.render()

  def preprocess(self, frame):
    frame = frame.astype(np.uint8)
    observation = cv2.resize(frame, self.observation_dims)
    return observation

  class Ball(DrawableRect):
    # Size of ball
    WIDTH = 10
    HEIGHT = 10
    X_SPEED = 3
    Y_SPEED = 2
    COLOR = WHITE

    def __init__(self):
      super(Ball, self).__init__(0, 0, Ball.WIDTH, Ball.HEIGHT, Ball.COLOR)
      self.reset()

    # Returns the a value for the reward as viewed from paddle1
    def update(self, paddle1, paddle2, screen_dims):
      assert (paddle1.pos != paddle2.pos)
      if (paddle1.pos == 'LEFT'):
        paddleLeft = paddle1
        paddleRight = paddle2
        score_multiplier = 1
      else:
        paddleLeft = paddle2
        paddleRight = paddle1
        score_multiplier = -1

      result = 0
      if (self._collision(paddleLeft, paddleRight)):
        self.x_direction = self.x_direction * -1
      if (self.y_direction == -1):
        if (self.y <= 0):
          self.y_direction = 1
      if (self.y_direction == 1):
        if (self.y + Ball.HEIGHT >= screen_dims[1]):
          self.y_direction = -1
      if (self.x <= 0):
        result = -1
        self.x_direction = 1
      if (self.x >= screen_dims[0]):
        result = 1
        self.x_direction = -1
      self.x += self.x_direction * Ball.X_SPEED
      self.y += self.y_direction * Ball.Y_SPEED
      return result * score_multiplier

    def reset(self, screen_dims):
      self.x = screen_dims[0] / 2 - Ball.WIDTH / 2
      self.y = screen_dims[1] / 2 - Ball.HEIGHT / 2
      self.x_direction = 2 * random.randint(0, 1) - 1
      self.y_direction = 2 * random.randint(0, 1) - 1

    def _collision(self, paddleLeft, paddleRight):
      if (self.x_direction == 1):
        if (self.x + Ball.WIDTH >= paddleRight.x and self.y >= paddleRight.y and
            self.y <= paddleRight.y + Paddle.HEIGHT):
          return True
      else:
        if (self.x <= paddleLeft.x + Paddle.WIDTH and self.y >= paddleLeft.y and
            self.y <= paddleLeft.y + Paddle.HEIGHT):
          return True
      return False

  class Paddle(DrawableRect):
    #Size of paddle
    WIDTH = 10
    HEIGHT = 60
    BUFFER = 5

    SPEED = 2

    COLOR = WHITE

    def __init__(self, pos):
      assert (pos == 'LEFT' or pos == 'RIGHT')
      super(Paddle, self).__init__(x, y, Paddle.WIDTH, Paddle.HEIGHT,
                                   Paddle.COLOR)
      self.y_velocity = 0
      self.pos = pos

    def setVelocity(self, new_vel):
      if (abs(new_vel) <= Paddle.SPEED):
        self.y_velocity = new_vel

    def update(self, action):
      self.setVelocity(action * Paddle.SPEED)
      self.y += self.y_velocity
      if (self.y <= 0):
        self.y = 0
      if (self.y + Paddle.HEIGHT >= WINDOW_HEIGHT):
        self.y = WINDOW_HEIGHT - Paddle.HEIGHT

    def reset(self, screen_dims, pos):
      if self.pos == 'LEFT':
        self.x = Paddle.BUFFER
        self.y = screen_dims[1] // 2
      elif self.pos == 'RIGHT':
        self.x = screen_dims[0] - Paddle.BUFFER
        self.y = screen_dims[1] // 2


class PongGame():

  def __init__(self):
    self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    self.paddleLeft = Paddle(Paddle.BUFFER,
                             WINDOW_HEIGHT / 2 - Paddle.HEIGHT / 2)
    self.paddleRight = Paddle(WINDOW_WIDTH - Paddle.BUFFER - Paddle.WIDTH,
                              WINDOW_HEIGHT / 2 - Paddle.HEIGHT / 2)
    self.ball = Ball(WINDOW_WIDTH / 2 - Ball.WIDTH / 2,
                     WINDOW_HEIGHT / 2 - Ball.HEIGHT / 2)
    self.tally = 0

  def getPresentFrame(self):
    pygame.event.pump()
    self.screen.fill(BLACK)
    self.paddleLeft.draw(self.screen)
    self.paddleRight.draw(self.screen)
    self.ball.draw(self.screen)
    # Get pixels
    image_data = pygame.surfarray.array3d(pygame.display.get_surface())
    #Update
    pygame.display.flip()
    #Return screen data
    return self.tally, image_data


if __name__ == "__main__":
  game = PongGame()
  for i in range(1000):
    game.Update([0, 0])
    time.sleep(1 / FPS)
