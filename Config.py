# this file is for the collection of parameters

import torch
import numpy as np
from numpy import pi
import pandas as pd
import datetime


class Config:
    def __init__(self):
        self.SEED_NUMBER = 0

        self.WORLD_SIZE = 1.0  # 2.5
        self.ACTION_DIM = 2
        self.STATE_DIM = 29 #4  # 20

        self.TERMINAL_VEL = 0.1  # norm(action) that you believe as a signal to stop 0.1

        # all times are in second
        self.DELTA_T = 0.01  # time to perform one action
        self.EPISODE_TIME = 1  # # maximum length of time for one episode. if monkey can't firefly within this time period, new firefly comes
        self.EPISODE_LEN = int(self.EPISODE_TIME / self.DELTA_T)  # number of time steps(actions) for one episode

        self.TOT_T = 200000  # total number of time steps for this code

        self.BATCH_SIZE = 64  # for replay memory (default:64)
        self.REWARD = 10  # for max reward
        self.NUM_EPOCHS = 2# for replay memory
        self.DISCOUNT_FACTOR = 0.9

        self.BOX_STEP_SIZE = 5e-1
        self.STD_STEP_SIZE = 1e-4  # 1e-4 action space noise (default: 2e-3)

        self.filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.gains_range = [8, 12, 13, 17] # [vel min, vel max, ang min, ang max]
        self.std_range = [1, 2, 1, 2]# [vel min, vel max, ang min, ang max]
        self.goal_radius_range = [0.35* self.WORLD_SIZE, 0.4* self.WORLD_SIZE] #0.375: best radius



