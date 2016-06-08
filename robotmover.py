from read_config import read_config
from copy import deepcopy
from random import random

class RobotMover:
  def __init__(self, map):
    self.location = (0,0)
    self.config = read_config()
    self.prob_move_forward = self.config["prob_move_forward"]
    self.prob_move_backward = self.config["prob_move_backward"]
    self.prob_move_left = self.config["prob_move_left"]
    self.prob_move_right = self.config["prob_move_right"]
    self.map = deepcopy(map)

  def take_action(self, h, w, action):
    fwd = 0 + self.prob_move_forward
    bck = fwd + self.prob_move_backward
    lft = bck + self.prob_move_left
    rgt = lft + self.prob_move_right
    
    select = random()
    if select >= 0 and select < fwd:
      select = fwd
    elif select >= fwd and select < bck:
      select = bck
    elif select >= bck and select < lft:
      select = lft
    else:
      select = rgt
    
    if action == 'up':
      if select == fwd:
        move_cord = (-1,0)
      elif select == bck:
        move_cord = (1,0)
      elif select == lft:
        move_cord = (0,-1)
      else:
        move_cord = (0,1)
        
    elif action == 'down':
      if select == fwd:
        move_cord = (1,0)
      elif select == bck:
        move_cord = (-1,0)
      elif select == lft:
        move_cord = (0,1)
      else:
        move_cord = (0,-1)

    elif action == 'left':
      if select == fwd:
        move_cord = (0,-1)
      elif select == bck:
        move_cord = (0,1)
      elif select == lft:
        move_cord = (1,0)
      else:
        move_cord = (-1,0)
        
    elif action == 'right':
      if select == fwd:
        move_cord = (0,1)
      elif select == bck:
        move_cord = (0,-1)
      elif select == lft:
        move_cord = (-1,0)
      else:
        move_cord = (1,0)

    (x,y)=(h+move_cord[0],w+move_cord[1])
    if (x<0 or  y<0 or  x>=len(self.map) or y>=len(self.map[0])):
      return (h,w)
    elif (self.map[x][y].is_wall):
      return (h,w)
    return (x,y)