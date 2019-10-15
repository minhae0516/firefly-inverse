# this file is for displaying play animation by loading pretrained ddpg network

import gym
from parameter import *
#from gym.wrappers.monitoring.video_recorder import VideoRecorder

import time
import numpy as np
from DDPGv2Agent import Agent, Noise
from collections import deque
rewards = deque(maxlen=100)
video_path = './pretrained/ddpg_minhae/video.mp4'

TOT_T = 500
env = gym.make('FireflyTorch-v0')
#rec = VideoRecorder(env, video_path, enabled=video_path is not None) #for video
state_dim = env.state_dim
action_dim = env.action_dim

std = 0.05
noise = Noise(action_dim, mean=0., std=std)
agent = Agent(PROC_NOISE_STD, OBS_NOISE_STD, gains, obs_gains, rew_std, state_dim, action_dim, hidden_dim=128, tau=0.001)
agent.load('pretrained/ddpg_minhae/ddpg_model_EE.pth.tar')

tot_t = 0.
episode = 0.
while tot_t <= TOT_T:
    episode += 1 # every episode starts a new firefly
    t, x, P, ox, b, state = env.reset()
    episode_reward = 0.

    while t < EPISODE_LEN:
        action = agent.select_action(state, noise)
        next_x, reached_target, next_b, reward, info, next_state = env.step(episode, x, b, action, t)
        env.Brender(next_b, next_x)  # display pop-up window (for every new action in each step)
        #rec.capture_frame() # for video
        time.sleep(0.1)  # delay for 0.005 sec
        if info['stop']:
            time.sleep(2)


        # check time limit
        TimeEnd = (t + 1 == EPISODE_LEN)  # if the monkey can't catch the firefly in EPISODE_LEN, reset the game.

        # update variables
        episode_reward += reward[0].item()
        x = next_x
        state = next_state
        b = next_b
        t += 1.
        tot_t += 1.
        if info['stop'] or TimeEnd:  # if the monkey stops or pass the time limit, start the new firefly
            break

    rewards.append(episode_reward / t)  # average reward per one time step
    avg_rew = np.mean(rewards)  # average reward for all episodes
    print("Ep: {}, steps: {}, EPrew: {:0.4f}, TOTavg_rew: {:0.4f}".format(episode, int(t), rewards[-1].item(),avg_rew))
#rec.close()
#rec.enabled = False
print("Done")

