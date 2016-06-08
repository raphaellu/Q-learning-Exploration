from read_config import *
from math import *
from copy import deepcopy
from random import random
from robotmover import *

CONSTANT_IN_L = 20
PRINT_ITER = False

class PGrid:
  def __init__(self):
    self.reward_list = { 'down': 0, 'left': 0, 'right': 0, 'up': 0}
    self.action_counters = { 'down': 0, 'left': 0, 'right': 0, 'up': 0}
    #default big number
    self.reward = 0
    self.policy = ""
    self.is_pit = False
    self.is_wall = False
    self.is_goal = False

  def __str__(self):
    return '"' + str(self.policy) + '"'

  def __repr__(self):
    return self.__str__()

class Learning:
  def __init__(self):
    self.config = read_config()
    self.move_list = self.config["move_list"]
    self.start = self.config["start"]
    self.goal = self.config["goal"]
    self.walls = self.config["walls"]
    self.pits = self.config["pits"]
    self.map_size = self.config["map_size"]
  
    
    # config data for mdp
    self.threshold_diff = self.config["threshold_difference"]
    self.max_iteration = self.config["max_iterations"]
    self.reward_step = self.config["reward_for_each_step"]
    self.reward_wall = self.config["reward_for_hitting_wall"]
    self.reward_goal = self.config["reward_for_reaching_goal"]
    self.reward_pit = self.config["reward_for_falling_in_pit"]
    self.discount_factor = self.config["discount_factor"]
    self.learning_factor = 0.001
    
    # initialize the map
    self.map = self.init_map_structure()
    self.robot_mover = RobotMover(self.map)

    self.compute_map_policy()
    self.policy_list = self.flatten_map(self.map)


  def flatten_map(self, map):
    return [x for sub in map for x in sub]

  def compute_map_policy(self):
    global PRINT_ITER
    #initialize the util_diff and num_iter
    util_diff = self.threshold_diff + 10;
    num_iter = 0
    prev_iter = [1]
    after_iter = [2]
    while util_diff > self.threshold_diff and num_iter < self.max_iteration:
      old_util = self.cal_map_util()
      if PRINT_ITER:
        prev_iter = self.flatten_map(self.map)
      self.compute_policy_iteration()
      if PRINT_ITER:
        after_iter = self.flatten_map(self.map)
        if repr(prev_iter) != repr(after_iter):
          print prev_iter
      new_util = self.cal_map_util()
      util_diff = abs(new_util - old_util)
      #print util_diff
      num_iter += 1

  def init_map_structure(self):
    # an empty map where each elmt is a tuple
    map = [ [ PGrid() for y in range(0, self.map_size[1]) ] for x in range(0, self.map_size[0]) ]
    # mark walls/pits/goal avoided boolean to True ( indicating to avoid )
    for wall in self.walls:
      map[wall[0]][wall[1]].is_wall = True
      map[wall[0]][wall[1]].reward = self.reward_wall
      map[wall[0]][wall[1]].policy = "WALL"
    for pit in self.pits:
      map[pit[0]][pit[1]].is_pit = True
      map[pit[0]][pit[1]].reward = self.reward_pit
      map[pit[0]][pit[1]].policy = "PIT"
      
    map[self.goal[0]][self.goal[1]].is_goal = True
    map[self.goal[0]][self.goal[1]].reward = self.reward_goal
    map[self.goal[0]][self.goal[1]].policy = "GOAL"
    
    return map

  def compute_policy_iteration (self):
    self.temp_map=deepcopy(self.map)
    
    for h in range(0, len(self.map)):
      for w in range(0, len(self.map[0])):
        grid = self.map[h][w]
        if not (grid.is_goal or grid.is_wall or grid.is_pit):
          self.compute_grid_policy(h,w)
          n_policy = ""
          n_reward = -99999
          # find the max reward in reward_list
          for act in self.temp_map[h][w].reward_list:
            act_r = self.temp_map[h][w].reward_list[act]
            if n_reward <= act_r:
              n_reward = act_r
              n_policy = act

          if n_policy == 'left':
            n_policy = 'W'
          elif n_policy == 'right':
            n_policy = 'E'
          elif n_policy == 'up':
            n_policy = 'N'
          elif n_policy == 'down':
            n_policy = "S"

          self.temp_map[h][w].policy = n_policy
          self.temp_map[h][w].reward = n_reward
    
    self.map = self.temp_map
  
  def compute_grid_policy (self,h,w):
    global CONSTANT_IN_L
    
    grid = self.map[h][w]
    temp_grid = self.temp_map[h][w]
    
    max_reward = -99999
    max_dir = "init"
    
    # decide which direction to go 
    for dir in temp_grid.reward_list:
      dir_r = temp_grid.reward_list[dir]
      if temp_grid.action_counters[dir] == 0:
        max_reward = dir_r
        max_dir = dir
        break
      elif max_reward <= dir_r + CONSTANT_IN_L/temp_grid.action_counters[dir]:
        max_reward = dir_r + CONSTANT_IN_L/temp_grid.action_counters[dir]
        max_dir = dir
    # print "//////"
    # print max_dir

    temp_grid.action_counters[max_dir] += 1
    
    # from state (h,w), take action "max_dir", land on state (x,y)
    (x,y) = self.robot_mover.take_action(h,w,max_dir)
    # print (x,y)

    self.update_Q_value(h,w,x,y,max_dir)
    # assume Q value has been updated
    
        
    
    
  def update_Q_value(self,x0,y0,x1,y1,action):
    # print str(x0) + "/" + str(y0) + "/" + str(x1) + "/" + str(y1) + "/" + action 
    curr_grid = self.map[x0][y0]
    land_grid = self.map[x1][y1]
 
    # Q_max_key =  max(land_grid.reward_list, key=land_grid.reward_list.get)
    # Q_max_value = self.discount_factor * land_grid.reward_list.get(Q_max_key, default=None)
    # for key in land_grid.reward_list:
      # print str(key) + " - " + str(land_grid.reward_list[key])
    Q_max_value = land_grid.reward
    # print str(Q_max_value) + " q max value"
    Q_sa = curr_grid.reward_list[action] 
    R_sa = self.reward_step
    Q_value = (1-self.learning_factor)*Q_sa + self.learning_factor*(R_sa + Q_max_value * self.discount_factor)    
    self.temp_map[x0][y0].reward_list[action] = Q_value
    # print str(Q_value) + " q value"

  def cal_map_util(self):
    tot = 0
    for h in range(0, len(self.map)):
      for w in range(0, len(self.map[0])):
        grid = self.map[h][w]
        if not (grid.is_goal or grid.is_wall or grid.is_pit):
          for r in grid.reward_list:
            tot += grid.reward_list[r]
    return tot

if __name__ == '__main__':
  l = Learning()
  print l.policy_list
  # for row in l.map:
    # print row


  
    

  
