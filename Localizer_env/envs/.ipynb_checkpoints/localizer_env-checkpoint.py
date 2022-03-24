#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 10:34:30 2022

@author: mstaps
"""
import numpy as np
import random
import math

import gym
from gym import wrappers
from gym import Env, spaces
from gym import utils
from gym.utils import seeding
from gym.spaces import Discrete, Box




class localizer_env(gym.Env):
    """
    A very simple environment to test agents & networks of more complex env
    
    Idea is to use the same observations as the more complex env
    """
    def __init__(self):
        super().__init__()
        
        HEIGHT, WIDTH = 200, 200
        
        self.HEIGHT, self.WIDTH = HEIGHT, WIDTH
        #*************************************************************************** 
        # Define Observation Spaces
        #*************************************************************************** 
        
        # vector observation [pos_x, pos_y, con, wind_x, wind_y]
        self.observation_space_1 = spaces.Box(low = np.array([-1.00,-1.00,0,-15,-15]),
                                              high=np.array([200,200,50,15,15]), 
                                              shape=(5,), 
                                              dtype=np.float64 )
        # image obs (2D) 
        self.observation_space_2 = spaces.Box(low=0.00,
                                              high=2.0,
                                              shape=(HEIGHT, WIDTH), 
                                              dtype=np.float64)
        
        self.observation_space = spaces.Dict({
        'position' : spaces.Box(low = -1.00,
                                high =  200,
                                shape=(2,),
                                dtype=np.float64),
        'concentration' : spaces.Box(low = 0.00,
                                     high=50,
                                     shape=(1,),
                                     dtype=np.float64),
        'wind' : spaces.Box(low = -15,
                            high = 15,
                            shape=(2,), 
                            dtype=np.float64),
        'Source Map': spaces.Box(low=0.00,
                                 high=2.0,
                                 shape=(HEIGHT, WIDTH), 
                                 dtype=np.float64)
        })
        #  action = [speed(m/s), direction(rad)]
        self.action_space = spaces.Box(low=np.array([-2,0]), high=np.array([2,np.pi/2]), shape=(2,))
        self.id = 'Local-v0'
        
        
        #map of distances
        self.dist_map = np.zeros((HEIGHT, WIDTH))
        self.max_episode_steps = 2000
        self.step_count = 0
        self.current_user_location = np.random.randint(low=[0,0],high=[200,200])
        self.goal_location = np.random.randint(low=[0,0],high=[200,200])
        self.done =False
        self.s = dict()
        self.s2 = dict()
        
    def step(self, desired_action):
        
        #***************************************************************************
        # Move Agent
        #*************************************************************************** 
        
        #Convert action into a position
        
        new_state = np.add(np.multiply(desired_action,
                                       np.array([np.cos(desired_action[1]),
                                                 np.sin(desired_action[1])])),
                           self.current_user_location)
        
        in_area_mask = ((new_state[0] >= 0) & 
                        (new_state[1] >= 0) & 
                        (new_state[0] <= self.HEIGHT) & 
                        (new_state[1] <= self.WIDTH))
        if(in_area_mask):
            self.current_user_location = new_state
            self.step_count = 1+ self.step_count
            self.done =False
        else:
            self.done = True
        #***************************************************************************
        # Get Measurements
        #***************************************************************************
        if not self.done:
            direction_vect = np.subtract(self.goal_location,
                                         self.current_user_location)
            # get distance from goal
            self.ConcOut = np.sqrt(np.sum(np.power(direction_vect,2)))
            #get direction to goal (normalized)
            self.WindRobot = np.divide(direction_vect,self.ConcOut)
        else:
            self.reward = -self.max_episode_steps
        #***************************************************************************
        # Calculate Reward
        #***************************************************************************
        if self.step_count<self.max_episode_steps:
            if self.ConcOut < 2:
                self.done = True
                self.reward = 10
            else:
                self.reward = -1
        #***************************************************************************
        # Observations
        #***************************************************************************            
        # Store the observation of the state
        self.s2 = {'obs1': np.array([self.current_user_location[0],
                                     self.current_user_location[1],
                                     self.ConcOut,
                                     self.WindRobot[0],
                                     self.WindRobot[1]]),
                   'obs2':self.dist_map
                   }
        
        #
        self.s = {'position':np.array(self.current_user_location),
                  'concentration':np.array([self.ConcOut]),
                  'Source Map':self.dist_map,
                  'wind':self.WindRobot}
        
        info = dict()
        
        return self.s2, self.reward, self.done, info
    def seed(self, seed=None):
        """ Set the random seed """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def reset(self):
        #map of distances
        #map of distances
        self.dist_map = np.zeros((self.HEIGHT, self.WIDTH))
        self.max_episode_steps = 2000
        self.step_count = 0
        self.current_user_location = np.random.randint(low=[0,0],high=[self.HEIGHT,self.WIDTH])
        self.goal_location = np.random.randint(low=[0,0],high=[self.HEIGHT,self.WIDTH])
        direction_vect = np.subtract(self.goal_location,
                                     self.current_user_location)
        # get distance from goal
        self.ConcOut = np.sqrt(np.sum(np.power(direction_vect,2)))
        #get direction to goal (normalized)
        self.WindRobot = np.divide(direction_vect,self.ConcOut)
        self.done =False
        self.s = dict()
        self.s2 = dict()
        self.s2 = {'obs1': np.array([self.current_user_location[0],
                             self.current_user_location[1],
                             self.ConcOut,
                             self.WindRobot[0],
                             self.WindRobot[1]]),
                   'obs2':self.dist_map
                  }
        
        
        return self.s2
        
