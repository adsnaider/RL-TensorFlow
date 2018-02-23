from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pygame
import random
import time

from environment.environment import Environment
from environment.view import View, DrawableRect

import numpy as np
import cv2

# RGB Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)  # Initialize our screen

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 640
screen_dims = [SCREEN_WIDTH, SCREEN_HEIGHT]
SCREEN_COLOR = BLACK
ACTIONS = 3  # Up, down, nothing


class Pong(Environment):

  def __init__(self, name, observation_dims, window_dims, rendering):
    super(Pong, self).__init__()
    self.actions = ACTIONS
    self.name = name
    self.observation_dims = np.array(observation_dims)
    self.window_dims = np.array(window_dims)
    self.window_ratio = self.window_dims / screen_dims
    self.rendering = rendering

    self.paddle_nn = Paddle('LEFT', self.window_ratio)  # DQN
    self.paddle_ai = Paddle('RIGHT', self.window_ratio)  # simple AI
    self.ball = Ball(self.window_ratio)
    self.draw_list = [self.paddle_nn, self.paddle_ai, self.ball]
    self.score = 0
    self.view = View(name, screen_dims, window_dims)

  def reset(self):
    self.paddle_nn.reset(self.view.screen_dims, 'LEFT')
    self.paddle_ai.reset(self.view.screen_dims, 'RIGHT')
    self.ball.reset(self.view.screen_dims)
    self.score = 0

  def new_game(self):
    self.reset()
    self.view.create_screen()
    self._update_view()
    return self.preprocess(self.view.get_current_frame()), self.score, False

  def step(self, action):
    # Update NN
    self.paddle_nn.update(action - 1)
    # Update AI
    action = 0
    if (self.paddle_ai.y + self.paddle_ai.HEIGHT / 2 <
        self.ball.y + self.ball.HEIGHT / 2):
      action = 1
    elif (self.paddle_ai.y + self.paddle_ai.HEIGHT / 2 >
          self.ball.y + self.ball.HEIGHT / 2):
      action = -1
    self.paddle_ai.update(action)  # AI will do 0, 1, 2

    point = self.ball.update(self.paddle_nn, self.paddle_ai,
                             self.view.screen_dims)
    self.score += point
    if (point != 0):
      terminal = True
      self.reset()
    else:
      terminal = False
    self._update_view()
    return self.preprocess(self.view.get_current_frame()), self.score, terminal

  def _update_view(self):
    self.view.update(self.draw_list, SCREEN_COLOR)
    if (self.rendering):
      self.view.render()

  def preprocess(self, frame):
    frame = frame.astype(np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    observation = cv2.resize(gray, tuple(self.observation_dims[:2]))
    return observation[..., np.newaxis]


class Ball(DrawableRect):
  # Size of ball
  WIDTH = 20
  HEIGHT = 20
  X_SPEED = 30
  Y_SPEED = 20
  COLOR = WHITE

  def __init__(self, window_ratio):
    super(Ball, self).__init__(0, 0, self.WIDTH, self.HEIGHT, self.COLOR)
    self.window_ratio = window_ratio

  # Returns the a value for the reward as viewed from paddle1
  def update(self, paddle1, paddle2, window_dims):
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
    value = self._collision(paddleLeft, paddleRight)
    if (value != 0):
      self.x_direction = value
    if (self.y_direction == -1):
      if (self.y <= 0):
        self.y_direction = 1
    if (self.y_direction == 1):
      if (self.y + self.HEIGHT >= window_dims[1]):
        self.y_direction = -1
    if (self.x + self.WIDTH <= 0):
      result = -1
    if (self.x >= window_dims[0]):
      result = 1
    self.x += self.x_direction * self.X_SPEED
    self.y += self.y_direction * self.Y_SPEED
    return result * score_multiplier

  def reset(self, screen_dims):
    self.x = screen_dims[0] // 2 - self.WIDTH // 2
    self.y = screen_dims[1] // 2 - self.HEIGHT // 2
    self.x_direction = 2 * random.randint(0, 1) - 1
    self.y_direction = 2 * random.randint(0, 1) - 1

  def _collision(self, paddleLeft, paddleRight):
    if (self.x_direction == 1):
      if (self.x + self.WIDTH >= paddleRight.x and self.y >= paddleRight.y and
          self.y <= paddleRight.y + Paddle.HEIGHT):
        return -1
    else:
      if (self.x <= paddleLeft.x + Paddle.WIDTH and self.y >= paddleLeft.y and
          self.y <= paddleLeft.y + Paddle.HEIGHT):
        return 1
    return 0


class Paddle(DrawableRect):
  #Size of paddle
  WIDTH = 10
  HEIGHT = 60
  BUFFER = 5

  SPEED = 18

  COLOR = WHITE

  def __init__(self, pos, window_ratio):
    assert (pos == 'LEFT' or pos == 'RIGHT')
    super(Paddle, self).__init__(0, 0, self.WIDTH, self.HEIGHT, self.COLOR)
    self.y_velocity = 0
    self.pos = pos
    self.window_ratio = window_ratio

  def setVelocity(self, new_vel):
    if (abs(new_vel) <= self.SPEED):
      self.y_velocity = new_vel

  def update(self, action):
    self.setVelocity(action * self.SPEED)
    self.y += self.y_velocity
    if (self.y <= 0):
      self.y = 0
    if (self.y + Paddle.HEIGHT >= SCREEN_HEIGHT):
      self.y = SCREEN_HEIGHT - self.HEIGHT

  def reset(self, screen_dims, pos):
    if self.pos == 'LEFT':
      self.x = self.BUFFER
      self.y = screen_dims[1] // 2
    elif self.pos == 'RIGHT':
      self.x = screen_dims[0] - self.BUFFER - self.WIDTH
      self.y = screen_dims[1] // 2


if __name__ == "__main__":
  pong = Pong('Pong', [20, 20],
              ([int(1.2 * SCREEN_WIDTH),
                int(1.4 * SCREEN_HEIGHT)]), True)
  pong.new_game()
  FPS = 60
  for i in range(1000):
    pong.step(1)
    time.sleep(1 / FPS)
