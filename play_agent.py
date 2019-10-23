# this file is for displaying play animation by loading pretrained ddpg network

import gym
#from gym.wrappers.monitoring.video_recorder import VideoRecorder
import time

import pandas as pd
from mplotter import *
from DDPGv2Agent import Agent, Noise
from collections import deque
rewards = deque(maxlen=100)

# read configuration parameters
from Config import Config
arg = Config()
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

import datetime
import pandas as pd

filename = '20191016-205855' # agent information
df = pd.read_csv('../firefly-inverse-data/data/' + filename + '_log.csv',
                 usecols=['discount_factor','process gain forward', 'process gain angular', 'process noise std forward',
                          'process noise std angular', 'obs gain forward', 'obs gain angular', 'obs noise std forward',
                          'obs noise std angular', 'goal radius'])

DISCOUNT_FACTOR = df['discount_factor'][0]
gains_range = [np.floor(df['process gain forward'].min()), np.ceil(df['process gain forward'].max()),
               np.floor(df['process gain angular'].min()), np.ceil(df['process gain angular'].max())]

std_range = [df['process noise std forward'].min(), df['process noise std forward'].max(),
               df['process noise std angular'].min(), df['process noise std angular'].max()]
goal_radius_range = [df['goal radius'].min(), df['goal radius'].max()]



env = gym.make('FireflyTorch-v0') #,PROC_NOISE_STD,OBS_NOISE_STD)
env.setup(arg)
x, b, state, pro_gains, pro_noise_stds, obs_gains, obs_noise_stds, goal_radius  = env.reset(gains_range, std_range, goal_radius_range)
state_dim = env.state_dim
action_dim = env.action_dim

MAX_EPISODE = 20
std = 0.00001 #0.05
noise = Noise(action_dim, mean=0., std=std)

agent = Agent(state_dim, action_dim, arg,  filename, hidden_dim=128, gamma=DISCOUNT_FACTOR, tau=0.001)
agent.load(filename)

tot_t = 0.
episode = 0.


COLUMNS = ['total time', 'ep', 'time step', 'reward', 'goal',
           'a_vel', 'a_ang', 'true_r',
           'r', 'rel_ang', 'vel', 'ang_vel',
           'vecL1','vecL2','vecL3','vecL4','vecL5','vecL6','vecL7','vecL8','vecL9','vecL10',
           'vecL11','vecL12','vecL13','vecL14','vecL15',
           'process gain forward', 'process gain angular', 'process noise std forward', 'process noise std angular',
           'obs gain forward', 'obs gain angular', 'obs noise std forward', 'obs noise std angular', 'goal radius',
           'box_size', 'discount_factor']

history = pd.DataFrame(columns=COLUMNS)

while episode <= MAX_EPISODE:
    episode += 1 # every episode starts a new firefly
    t = torch.zeros(1) # to track the amount of time steps to catch a firefly

    theta = (pro_gains, pro_noise_stds, obs_gains, obs_noise_stds, goal_radius)
    env.Brender(b, x, arg.WORLD_SIZE, goal_radius)  # display pop-up window (for every new action in each step)

    while t < arg.EPISODE_LEN: # for a single FF
        action = agent.select_action(state, action_noise = noise, param = None)  # with action noise

        next_x, reached_target, next_b, reward, info, next_state = env.step(episode, x, b, action, t, theta, arg.REWARD)
        env.Brender(next_b, next_x, arg.WORLD_SIZE, goal_radius)  # display pop-up window (for every new action in each step)

        #time.sleep(0.1)  # delay for 0.005 sec
        if info['stop']:
            time.sleep(1)
        # check time limit
        TimeEnd = (t+1 == arg.EPISODE_LEN) # if the monkey can't catch the firefly in EPISODE_LEN, reset the game.
        mask = torch.tensor([1 - float(TimeEnd)]) # mask = 0: episode is over

        data = np.array([[tot_t, episode,  t, reward,
                          reached_target.item(),action[0][0].item(), action[0][1].item(), torch.norm(x.view(-1)[0:2]).item(),
                          state[0][0].item(), state[0][1].item(), state[0][2].item(), state[0][3].item(),
                          state[0][5].item(), state[0][6].item(), state[0][7].item(), state[0][8].item(),
                          state[0][9].item(),
                          state[0][10].item(), state[0][11].item(), state[0][12].item(), state[0][13].item(),
                          state[0][14].item(),
                          state[0][15].item(), state[0][16].item(), state[0][17].item(), state[0][18].item(),
                          state[0][19].item(),
                          pro_gains[0].item(), pro_gains[1].item(), pro_noise_stds[0].item(), pro_noise_stds[1].item(),
                          obs_gains[0].item(), obs_gains[1].item(), obs_noise_stds[0].item(), obs_noise_stds[1].item(),
                          goal_radius.item(),
                          arg.WORLD_SIZE, DISCOUNT_FACTOR]])

        df1 = pd.DataFrame(data, columns=COLUMNS)
        history = history.append(df1)

        if info['stop'] or TimeEnd:  # if the monkey stops or pass the time limit, start the new firefly
            next_x, next_b, next_state, pro_gains, pro_noise_stds, obs_gains, obs_noise_stds, goal_radius = env.reset(gains_range,
                                                                                                       std_range,
                                                                                                       goal_radius_range)

        # update variables
        x = next_x
        state = next_state
        b = next_b
        t += 1.
        tot_t += 1.

        if info['stop'] or TimeEnd: # if the monkey stops or pass the time limit, start the new firefly
            break

history.to_csv(path_or_buf='../firefly-inverse-data/data/' + filename + '_history.csv', index=False)

#
#
# # plot radius vs <# of time steps>
# timestep_radius_plotter(filename, history['episode'].values, history['t'].values, np.stack(history['true location'].values), PROC_NOISE_STD[0].item(), OBS_NOISE_STD[0].item())
#
# # plot belief location
# location_plotter_arrow(filename, history['episode'].values, np.stack(history['belief location'].values), PROC_NOISE_STD[0].item(), OBS_NOISE_STD[0].item())
# location_plotter(filename, history['episode'].values, np.stack(history['belief location'].values),np.stack(history['true location'].values), PROC_NOISE_STD[0].item(), OBS_NOISE_STD[0].item())
# location_plotter2(filename, history['episode'].values, np.stack(history['belief location'].values),np.stack(history['true location'].values), PROC_NOISE_STD[0].item(), OBS_NOISE_STD[0].item())
#
# """
# # plot true location
# location_plotter_arrow(filename, history['episode'].values, np.stack(history['true location'].values), PROC_NOISE_STD.item(), OBS_NOISE_STD.item())
# location_plotter(filename, history['episode'].values, np.stack(history['true location'].values), PROC_NOISE_STD.item(), OBS_NOISE_STD.item())
# location_plotter2(filename, history['episode'].values, np.stack(history['true location'].values), PROC_NOISE_STD.item(), OBS_NOISE_STD.item())
# """
#
# # plot action
# action_plotter_csv(filename, history['action'].values, PROC_NOISE_STD[0].item(), OBS_NOISE_STD[0].item())
# action_time_plotter_csv(filename, history['action'].values, PROC_NOISE_STD[0].item(), OBS_NOISE_STD[0].item())
# #action_plotter_csv_sparse(filename, history['action'].values, PROC_NOISE_STD.item(), OBS_NOISE_STD.item(), 5)
#
# #rec.close()
# #rec.enabled = False



print("Done")
