"""
This is the main description for firefly task model
This code is for environment

This code uses the polar coordinate
next_x = torch.stack((vel, ang_vel, r, ang))
state = torch.cat([vel, ang_vel, r, ang, vecL, time]) # for policy network
"""

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from .env_utils import *
from .plotter_gym import Render

def dynamics(x, a, dt, box, pro_gains, pro_noise_stds):
    # dynamics
    px, py, ang, vel, ang_vel = torch.split(x.view(-1), 1)

    a_v = a[0]  # action for velocity
    a_w = a[1]  # action for angular velocity

    w = pro_noise_stds * torch.randn(2) # random process noise for [vel, ang_vel]

    vel = 0.0 * vel + pro_gains[0] * a_v + w[0]
    ang_vel = 0.0 * ang_vel + pro_gains[1] * a_w + w[1]
    ang = ang + ang_vel * dt
    ang = range_angle(ang) # adjusts the range of angle from -pi to pi

    px = px + vel * torch.cos(ang) * dt
    py = py + vel * torch.sin(ang) * dt
    px = torch.clamp(px, -box, box)
    py = torch.clamp(py, -box, box)
    next_x = torch.stack((px, py, ang, vel, ang_vel))

    return next_x.view(1,-1)

class Model(nn.Module):
    def __init__(self, arg):
        super(self.__class__, self).__init__()
        # constants
        self.dt = arg. DELTA_T
        self.action_dim = arg.ACTION_DIM
        self.state_dim = arg.STATE_DIM
        #self.goal_radius = arg.GOAL_RADIUS
        self.terminal_vel = arg.TERMINAL_VEL
        self.episode_len = arg.EPISODE_LEN
        self.episode_time = arg.EPISODE_LEN * self.dt
        self.box = arg.WORLD_SIZE #initial value
        self.max_goal_radius = arg.goal_radius_range[0]
        self.GOAL_RADIUS_STEP = arg.GOAL_RADIUS_STEP_SIZE
        #self.rendering = Render()
        #self.reset()

    def reset(self, gains_range, std_range, goal_radius_range, goal_radius=None, pro_gains=None, pro_noise_stds=None):

        self.pro_gains = torch.zeros(2)
        self.pro_noise_stds = torch.zeros(2)


        if pro_gains is None:
            self.pro_gains[0] = torch.zeros(1).uniform_(gains_range[0], gains_range[1])  #[proc_gain_vel]
            self.pro_gains[1] = torch.zeros(1).uniform_(gains_range[2],
                                                        gains_range[3])  # [proc_gain_ang]
        else:
            self.pro_gains = pro_gains

        if pro_noise_stds is None:
            self.pro_noise_stds[0] = torch.zeros(1).uniform_(std_range[0], std_range[1]) #[proc_vel_noise]
            self.pro_noise_stds[1] = torch.zeros(1).uniform_(std_range[2],
                                                             std_range[3])  # [proc_ang_noise]
        else:
            self.pro_noise_stds = pro_noise_stds

        if goal_radius is None:
            self.max_goal_radius = min(self.max_goal_radius + self.GOAL_RADIUS_STEP, goal_radius_range[1])
            #self.goal_radius = torch.zeros(1).uniform_(self.min_goal_radius, goal_radius_range[1])
            self.goal_radius = torch.zeros(1).uniform_(goal_radius_range[0], self.max_goal_radius)
        else:
            self.goal_radius = goal_radius


        self.time = torch.zeros(1)
        min_r = self.goal_radius.item()
        #if self.world_size > 1.0:
        #    self.box = min(self.world_size, self.box + self.BOX_STEP_SIZE)
        r = torch.zeros(1).uniform_(min_r, self.box)  # GOAL_RADIUS, self.box


        #min_r = 0 #self.goal_radius.item()
        ##if self.box > 1.0:
            #min_r = self.box - BOX_STEP_SIZE
        #r = torch.zeros(1).uniform_(min_r, self.box) # GOAL_RADIUS, self.box

        loc_ang = torch.zeros(1).uniform_(-pi, pi) # location angel: to determine initial location
        px = r * torch.cos(loc_ang)
        py = r * torch.sin(loc_ang)
        rel_ang = torch.zeros(1).uniform_(-pi/4, pi/4)
        ang = rel_ang + loc_ang + pi # heading angle of monkey, pi is added in order to make the monkey toward firefly
        ang = range_angle(ang)

        vel = torch.zeros(1)
        ang_vel = torch.zeros(1)

        x = torch.cat([px, py, ang, vel, ang_vel])
        return x, self.pro_gains, self.pro_noise_stds, self.goal_radius


    def forward(self, x, a):
        # get a real next state of monkey
        next_x = dynamics(x, a, self.dt, self.box, self.pro_gains, self.pro_noise_stds)
        pos = next_x.view(-1)[:2]
        reached_target = (torch.norm(pos) <= self.goal_radius)

        return next_x, reached_target

    """
    def render(self, x, P):
        goal = torch.zeros(2)
        self.rendering.render(goal, x, P)
        """

    def get_position(self, x):
        pos = x.view(-1)[:2]
        r = torch.norm(pos).item()
        return pos, r

    def input(self, x, obs_gains = None): # input to animal - no noise

        if obs_gains is None:
            obs_gains = self.obs_gains
        vel, ang_vel = torch.split(x.view(-1),1)[-2:]

        ovel = obs_gains[0] * vel
        oang_vel = obs_gains[1] * ang_vel
        ox = torch.stack((ovel, oang_vel))
        return ox
