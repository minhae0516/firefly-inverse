from .agent import Agent
from .noise import Noise, OUNoise

# these are for gym
from gym.envs.registration import register
#from .gym_input import true_params

register(
    id ='FireflyTorch-v0',
    entry_point ='FireflyEnv.firefly_gym:FireflyEnv',
    #entry_point='DDPGv2Agent.firefly_gym:FireflyEnv',

)
"""
import sys
sys.path.append("..")
from FireflyEnv.gym_input import true_params
"""