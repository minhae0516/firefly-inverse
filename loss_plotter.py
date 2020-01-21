import torch
import torch.nn as nn
from torch.autograd import grad
import pandas as pd
from InverseFuncs import trajectory, getLoss, reset_theta, theta_range

from DDPGv2Agent import Agent
from FireflyEnv import Model # firefly_task.py
from collections import deque
from Inverse_Config import Inverse_Config
import matplotlib.pyplot as plt

# read configuration parameters
arg = Inverse_Config()
# fix random seed
import random
random.seed(arg.SEED_NUMBER)
import torch
torch.manual_seed(arg.SEED_NUMBER)
if torch.cuda.is_available():
    torch.cuda.manual_seed(arg.SEED_NUMBER)
import numpy as np
np.random.seed(arg.SEED_NUMBER)
import time

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm






def loss_cal(vel, pro_gains_vel, gain_space, true_theta):
    loss_log = torch.zeros(len(gain_space) + 1)
    pro_gains = torch.zeros(2)
    pro_noise_stds = torch.zeros(2)
    obs_gains = torch.zeros(2)
    obs_noise_stds = torch.zeros(2)
    goal_radius = torch.zeros(1)

    theta_log = []
    inputs = np.sort(np.append(gain_space, true_theta[1]))

    pro_gains[0] = pro_gains_vel
    for ang, pro_gains_ang in enumerate(tqdm(inputs)):
        pro_gains[1] = pro_gains_ang
        for obs_gains_vel in [true_theta[4]]:  # np.append(gain_space, true_theta[4]):
            obs_gains[0] = obs_gains_vel
            for obs_gains_ang in [true_theta[5]]:  # np.append(gain_space, true_theta[5]):
                obs_gains[1] = obs_gains_ang
                for pro_std_vel in [true_theta[2]]:  # np.append(std_space, true_theta[2]):
                    pro_noise_stds[0] = pro_std_vel
                    for pro_std_ang in [true_theta[3]]:  # np.append(std_space, true_theta[3]):
                        pro_noise_stds[1] = pro_std_ang
                        for obs_std_vel in [true_theta[6]]:  # np.append(std_space, true_theta[6]):
                            obs_noise_stds[0] = obs_std_vel
                            for obs_std_ang in [true_theta[7]]:  # np.append(std_space, true_theta[7]):
                                obs_noise_stds[1] = obs_std_ang
                                for goal_r in [true_theta[8]]:  # np.append(goal_radius_space, true_theta[8]):
                                    goal_radius[0] = goal_r

                                    theta = torch.cat(
                                        [pro_gains, pro_noise_stds, obs_gains, obs_noise_stds, goal_radius])

                                    theta_log.append(theta.data)
                                    loss = getLoss(agent, x_traj, a_traj, theta, env, arg.gains_range, arg.std_range,
                                                   arg.PI_STD, arg.NUM_SAMPLES)
                                    loss_log[ang] = loss.data
                                    # loss_log.append(loss.data)

                                    print("num:{}, theta:{}, loss:{}".format(vel, theta, loss))

    """
    result = {'true_theta': true_theta,
         'true_loss': true_loss,
         'theta_log': theta_log,
         'loss_log': loss_log}
    """
    return loss_log


if __name__ == "__main__":

    num_cores = multiprocessing.cpu_count()
    print("{} cores are available".format(num_cores))

    # if gpu is to be used
    #CUDA = False
    #device = "cpu"

    CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tic = time.time()



    filename = '20191231-172726-01081157' # agent information

    learning_arg = torch.load('../firefly-inverse-data/data/20191231-172726_arg.pkl')

    DISCOUNT_FACTOR = learning_arg.DISCOUNT_FACTOR
    arg.gains_range = learning_arg.gains_range
    arg.std_range = learning_arg.std_range
    arg.goal_radius_range = learning_arg.goal_radius_range
    arg.WORLD_SIZE = learning_arg.WORLD_SIZE
    arg.DELTA_T = learning_arg.DELTA_T
    arg.EPISODE_TIME = learning_arg.EPISODE_TIME
    arg.EPISODE_LEN = learning_arg.EPISODE_LEN



    env = Model(arg) # build an environment
    env.max_goal_radius = arg.goal_radius_range[1] # use the largest world size for goal radius
    env.box = arg.WORLD_SIZE
    agent = Agent(env.state_dim, env.action_dim, arg,  filename, hidden_dim=128, gamma=DISCOUNT_FACTOR, tau=0.001) #, device = "cpu")
    agent.load(filename)

    # true theta
    true_theta = reset_theta(arg.gains_range, arg.std_range, arg.goal_radius_range)
    x_traj, obs_traj, a_traj, _ = trajectory(agent, true_theta, env, arg, arg.gains_range, arg.std_range,arg.goal_radius_range, arg.NUM_EP)  # generate true trajectory
    true_loss = getLoss(agent, x_traj, a_traj, true_theta, env, arg.gains_range, arg.std_range, arg.PI_STD, arg.NUM_SAMPLES)  # this is the lower bound of loss?
    print("true loss:{}".format(true_loss))
    print("true_theta:{}".format(true_theta))

    gain_space = np.linspace(arg.gains_range[0],arg.gains_range[1], num = 9)
    std_space = np.linspace(arg.std_range[0], arg.std_range[1], num = 3)
    goal_radius_space = np.linspace(arg.goal_radius_range[0], arg.goal_radius_range[1], num =3)



    inputs = np.sort(np.append(gain_space, true_theta[0]))
    #loss_log = torch.zeros([len(gain_space)+1, len(gain_space)+1])
    loss_log = Parallel(n_jobs=num_cores)(delayed(loss_cal)(vel, pro_gains_vel, gain_space, true_theta) for vel, pro_gains_vel in enumerate(inputs))

    loss_log_tot = torch.cat([loss_log[i] for i in range(len(loss_log))])
