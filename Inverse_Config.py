# this file is for the collection of parameters

import torch
import numpy as np
from numpy import pi
import pandas as pd
import datetime


class Inverse_Config:
    def __init__(self):
        self.SEED_NUMBER = 1004

        self.WORLD_SIZE = 1.0  # 2.5
        self.ACTION_DIM = 2
        self.STATE_DIM = 29 #4  # 20
        #self.GOAL_RADIUS = 0.375  * self.WORLD_SIZE  # 0.4 #0.5
        self.TERMINAL_VEL = 0.1  # norm(action) that you believe as a signal to stop 0.1

        # all times are in second
        #self.DELTA_T = 0.01  # time to perform one action
        #self.EPISODE_TIME = 1  # # maximum length of time for one episode. if monkey can't firefly within this time period, new firefly comes
        #self.EPISODE_LEN = int(self.EPISODE_TIME / self.DELTA_T)  # number of time steps(actions) for one episode

        self.TOT_T = 1000000000  # total number of time steps for this code

        #self.INVERSE_BATCH_SIZE = 300  # the number of actions for a set of trajectory
        self.REWARD = 10  # for max reward
        self.BOX_STEP_SIZE = 0
        self.GOAL_RADIUS_STEP_SIZE = 0
        self.PI_STD = 0.1 #1/5 # policy std --> just normalizer in this code
        self.NUM_SAMPLES = 50 # number of particles
        self.NUM_EP = 500
        self.NUM_IT = 60 # number of iteration for gradient descent
        self.NUM_thetas = 15

        self.ADAM_LR = 5e-1 #for learning rate annealing#5e-3
        self.LR_STEP = 1
        self.LR_STOP = 50
        self.lr_gamma = 0.95


        #self.action_vel_weight = 1
        #self.action_ang_weight = 1
        self.monkey_filename = '../firefly-inverse-data/monkey_traj.csv'

        #self.gains_range = [8, 12, 13, 17]  # [vel min, vel max, ang min, ang max]
        #self.std_range = [0, 1, 0, 1]  # [vel min, vel max, ang min, ang max]
        #self.goal_radius_range = [0.3 * self.WORLD_SIZE, 0.45 * self.WORLD_SIZE] #0.374




