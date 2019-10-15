from .firefly_task import Model
from .firefly_task import dynamics

#from .env_utils import pos_init
from .env_utils import *
#from .env_variables import *

"""
# these are for gym
from .gym_input import true_params
from gym.envs.registration import register
register(
    id ='FireflyTorch-v0',
    #entry_point ='FireflyEnv.firefly_gym:FireflyEnv',
    entry_point='firefly_gym:FireflyEnv',

)
"""