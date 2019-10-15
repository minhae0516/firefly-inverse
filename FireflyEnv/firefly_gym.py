import gym
from numpy import pi
import numpy as np
from gym import spaces
from gym.utils import seeding
from FireflyEnv.firefly_task import Model
from DDPGv2Agent.belief_step import BeliefStep
from FireflyEnv.plotter_gym import Render
from parameter import *
from DDPGv2Agent.rewards import *




class FireflyEnv(gym.Env): #, proc_noise_std = PROC_NOISE_STD, obs_noise_std =OBS_NOISE_STD):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        #super(self.__class__, self).__init__()
        low = np.append([0., -pi, -1., -1., 0.], -10*np.ones(16))
        high = np.append([10., pi, 1., 1., 100.], 10*np.ones(16))
        #low[-1], high[-1] = 0., 100.
        self.action_space = spaces.Box(-np.ones(2), np.ones(2), dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

    def setup(self, arg):

        self.model = Model(arg) # environment
        self.belief = BeliefStep(arg) # agent
        self.action_dim = self.model.action_dim
        self.state_dim = self.model.state_dim
        #self.reset(gains_range, std_range, goal_radius_range)
        self.rendering = Render()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, episode, x, b, action, t, theta, REWARD, finetuning=0):
        next_x, reached_target = self.model(x, action.view(-1))  # track true next_x of monkey
        ox = self.belief.observations(next_x)  # observation
        next_b, info = self.belief(b, ox, action, self.model.box)  # belief next state, info['stop']=terminal # reward only depends on belief
        next_state = self.belief.Breshape(next_b, t, theta)  # state used in policy is different from belief
        reward = return_reward(episode, info, reached_target, next_b, self.model.goal_radius, REWARD, finetuning)


        """
        if reached_target == 1:
            _, r = self.model.get_position(next_x)
            _, br = self.belief.get_position(next_b)
            print("reached target! belief radius {}, real radius: {}".format(br, r))
            print("==")
        """

        # reward
        #reward = return_reward(episode, info, reached_target, b)
        """
        if info['stop'] and reached_target:  # receive reward only if monkey stops & arrives at target
            reward = self.belief.get_reward(b)
            print("Goal!!, reward= %.3f" % reward.view(-1))
        else:
            reward = -0 * torch.ones(1)
        """

        return next_x, reached_target, next_b, reward, info, next_state

    def reset(self, gains_range, std_range, goal_radius_range):
        #time = torch.zeros(1)  # to track the amount of time steps to catch a firefly
        #x = self.model.reset(init).view(1, -1)
        x, pro_gains, pro_noise_stds, goal_radius = self.model.reset(gains_range, std_range, goal_radius_range)
        #P, ox, b, state = self.belief.reset(x, t)

        b, state, obs_gains, obs_noise_stds = self.belief.reset(x, torch.zeros(1) , pro_gains, pro_noise_stds, goal_radius, gains_range, std_range)

        return x, b, state, pro_gains, pro_noise_stds, obs_gains, obs_noise_stds, goal_radius

    def Brender(self, b, x, WORLD_SIZE, GOAL_RADIUS):
        bx, P = b
        goal = torch.zeros(2)
        self.rendering.render(goal, bx.view(1,-1), P, x.view(1,-1), WORLD_SIZE, GOAL_RADIUS)
    """
    def Brender(self, b, mode='human'):
        self.belief.render(b)

    def render(self, x, P, mode='human'):
        self.model.render(x, P)
    """

    def render(self, mode='human'):
        self.rendering.render(mode)

    def get_position(self, x):
        pos = x.view(-1)[:2]
        r = torch.norm(pos).item()
        return pos, r
