# this file is for the collection of parameters

import torch
import numpy as np
from numpy import pi
import pandas as pd



class Config:

    def __init__(self):
        SEED_NUMBER = 0

        WORLD_SIZE = 1.0  # 2.5
        ACTION_DIM = 2
        STATE_DIM = 4  # 20
        GOAL_RADIUS = 0.375 * WORLD_SIZE  # 0.4 #0.5
        TERMINAL_VEL = 0.1  # norm(action) that you believe as a signal to stop 0.1

        # all times are in second
        DELTA_T = 0.01  # time to perform one action
        EPISODE_TIME = 1  # # maximum length of time for one episode. if monkey can't firefly within this time period, new firefly comes
        EPISODE_LEN = int(EPISODE_TIME / DELTA_T)  # number of time steps(actions) for one episode

        # rew_std = GOAL_RADIUS/2 * torch.ones(1)  #std of Gaussian distribution for reward: 68%
        # rew_std = GOAL_RADIUS/2/2 * torch.ones(1)  #2*std of Gaussian distribution for reward: 95%

        # Switch for functions:
        ACTION_NOISE = True  # args.action# action space noise for exploration
        COS_ACTION_NOISE = False  # args.COS
        PARAM_NOISE = False  # args.param # parameter space noise for exploration
        FINETUNING = False  # updated based on fixed reward

        TOT_T = 1000000000  # total number of time steps for this code

        # RSNR = 5#0.01*10**3 #Root of signal to noise ratio (without dB) = gain/noise std (default: 3*10**3)
        # SNR_dB = 10*np.log10(RSNR**2)
        SNR_dB = 50
        RSNR = np.sqrt(10 ** (SNR_dB / 10))

        gains = torch.Tensor(
            [10., pi / 20 * 100])  # torch.Tensor([10., pi/20 * 100])# default: torch.Tensor([20., pi/6 * 100])
        obs_gains = torch.Tensor(
            [10., pi / 20 * 100])  # torch.Tensor([10., pi/20 * 100])#default: torch.Tensor([20., pi/6 * 100])

        PROC_NOISE_STD = gains / RSNR
        OBS_NOISE_STD = obs_gains * gains / RSNR
        # PROC_NOISE_STD = 1e-4 * torch.ones(2) # process noise std 1e-2
        # PROC_NOISE_STD = torch.Tensor([0.01, 0.01]) #1e-2 * torch.ones(2) # observation noise std 1e-2
        # OBS_NOISE_STD = torch.Tensor([1, 1.571])

        # OBS_NOISE_STD = gains/RSNR

        ###true_params = (PROC_NOISE_STD, OBS_NOISE_STD, gains, obs_gains, rew_std)

        BATCH_SIZE = 64  # for replay memory (default:64)
        REWARD = 10  # for max reward
        # NUM_EPOCHS = 2 # for replay memory
        DISCOUNT_FACTOR = 0.8

        BOX_STEP_SIZE = 5e-1
        if COS_ACTION_NOISE:
            STD_STEP_SIZE = 2e-2  # action space noise (default: 2e-3)
        else:
            STD_STEP_SIZE = 1e-4  # 1e-4 action space noise (default: 2e-3)

        # parameter space noise
        INITIAL_STDDEV = 0.06
        DESIRED_ACTION_STDDEV = 0.03  # (default: 0.03)
        ADAPTATION_COEFFICIENT = 1.01  # (default: 1.01)
        PARAM_NOISE_ADAPT_INTERVAL = 50

    def env_arg(self):
        arg = (DELTA_T, ACTION_DIM, STATE_DIM, GOAL_RADIUS, TERMINAL_VEL, EPISODE_LEN, WORLD_SIZE)
        return arg


