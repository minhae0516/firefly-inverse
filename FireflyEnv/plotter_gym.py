# this file is for gym visualization

import gym
import torch
from .env_utils import *
from gym.envs.classic_control import rendering
import matplotlib.pyplot as plt
#from parameter import *
import numpy as np


AGENT_SIZE, GOAL_SIZE = 5, 10

GREEN = (0.6274, 0.8313, 0.4078)
ORANGE = (0.9882,  0.4313,  0.3176)
GREY = (0.5, 0.5, 0.5)
GOLD = (255/255, 223/255, 0)

class Plotter():
    def __init__(self):
        self.trajectory = None
        return

    def plot(self, x):
        if self.trajectory is None:
            self.trajectory = x.view(1, -1)
        else:
            self.trajectory = torch.cat([self.trajectory, x.view(1, -1)])

    def show(self):
        x = self.trajectory[:, 0].data.numpy()
        y = self.trajectory[:, 1].data.numpy()
        plt.plot(x, y)
        plt.show()

def translate(point, center, scale):
    return (point - center) * scale

class Render(gym.Env):
    def __init__(self):
        self.viewer = None

    def render(self, goal_pos, x, P, sx, WORLD_SIZE, GOAL_RADIUS):
        goal_pos = goal_pos.detach().numpy()
        x = x.detach().numpy()
        P = P.detach().numpy()
        sx = sx.detach().numpy()

        screen_width, screen_height = 500, 500
        xyBox = WORLD_SIZE
        world_width = 2 * xyBox
        scale = screen_width/world_width

        center = translate(torch.zeros(2),  -xyBox, scale)
        goal_pos = translate(goal_pos, -xyBox, scale)

        agent_size = AGENT_SIZE
        goal_size = GOAL_SIZE

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

            goal = rendering.make_circle(goal_size, res=30)
            goal_ring = rendering.make_circle(GOAL_RADIUS * scale, res=30, filled=False)

            goal.set_color(*GREEN) #green
            goal_ring.set_color(*GREEN) #green

            self.goal_motion = rendering.Transform(translation=center)
            goal.add_attr(self.goal_motion)
            goal_ring.add_attr(self.goal_motion)

            agent = rendering.make_circle(agent_size, res=30)
            head = rendering.make_polygon([(0,5),(0,-5),(5,0)])
            star = rendering.make_polygon([(0, -10), (6, 8), (-10, -3.0), (10, -3.0), (-6, 8)])

            agent.set_color(*ORANGE) #orange
            head.set_color(*GREY)
            head.add_attr(rendering.Transform(translation=(10,0))) #offset
            star.set_color(*GOLD)

            self.agent_motion = rendering.Transform(translation=(0,0))
            self.head_motion = rendering.Transform() # for rotation (turning)
            self.star_motion = rendering.Transform(translation=(0,0))

            agent.add_attr(self.agent_motion)
            head.add_attr(self.head_motion)
            head.add_attr(self.agent_motion)
            star.add_attr(self.star_motion)

            self.viewer.add_geom(goal)
            self.viewer.add_geom(goal_ring)
            self.viewer.add_geom(star)
            self.viewer.add_geom(agent)
            self.viewer.add_geom(head)
            self.viewer.add_geom(agent) # dummy addition to replace with cov


        self.goal_motion.set_translation(goal_pos[0], goal_pos[1])
        # belief
        #r, ang = x[0][2], x[0][3]
        #px = r * np.cos(ang)
        #py = r * np.sin(ang)
        #position = np.array([px, py])
        position, ang = x[0][:2], x[0][2]
        move = translate(position, -xyBox, scale)
        self.agent_motion.set_translation(move[0], move[1])

        self.head_motion.set_rotation(ang)

        #star
        #sr, sang = sx[0][2], sx[0][3]
        #spx = sr * np.cos(sang)
        #spy = sr * np.sin(sang)
        #sposition = np.array([spx, spy])
        sposition, sang = sx[0][:2], sx[0][2]
        smove = translate(sposition, -xyBox, scale)
        self.star_motion.set_translation(smove[0], smove[1])

        #pts = np.vstack(ellipse(np.zeros(2), P[-2:,-2:], conf_int=5.991*scale**2)).T #100 added
        pts = np.vstack(ellipse(np.zeros(2), P[:2, :2], conf_int=5.991 * scale ** 2)).T  # 100 added
        pts = [tuple(v) for v in pts]

        cov = rendering.make_polygon(pts, False)
        cov.set_color(*ORANGE)
        cov.add_attr(rendering.Transform(translation=(move[0], move[1])))
        self.viewer.geoms[-1] = cov

        return self.viewer.render(return_rgb_array=False)
