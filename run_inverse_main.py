#import torch
#import torch.nn as nn
#from torch.autograd import grad
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm

from InverseFuncs import trajectory, getLoss, reset_theta, theta_range
from single_inverse import single_inverse

from DDPGv2Agent import Agent
from FireflyEnv import Model # firefly_task.py
from Inverse_Config import Inverse_Config

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

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# if gpu is to be used
CUDA = False
device = "cpu"

#CUDA = torch.cuda.is_available()
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





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
agent = Agent(env.state_dim, env.action_dim, arg,  filename, hidden_dim=128, gamma=DISCOUNT_FACTOR, tau=0.001, device = "cpu")
agent.load(filename)




true_theta_log = []
true_loss_log = []
final_theta_log = []
stderr_log = []
result_log = []
x_traj_log=[]
a_traj_log=[]

for num_thetas in range(3):

    # true theta
    true_theta = reset_theta(arg.gains_range, arg.std_range, arg.goal_radius_range)
    true_theta_log.append(true_theta.data.clone())
    x_traj, obs_traj, a_traj, _ = trajectory(agent, true_theta, env, arg, arg.gains_range, arg.std_range,
                                             arg.goal_radius_range, arg.NUM_EP)  # generate true trajectory
    true_loss = getLoss(agent, x_traj, a_traj, true_theta, env, arg.gains_range, arg.std_range, arg.PI_STD,
                        arg.NUM_SAMPLES)  # this is the lower bound of loss?

    x_traj_log.append(x_traj)
    a_traj_log.append(a_traj)
    true_loss_log.append(true_loss)

    print("true_theta:{}".format(true_theta_log))
    print("true loss:{}".format(true_loss_log))



num_cores = multiprocessing.cpu_count()
print("{} cores are available".format(num_cores))
inputs = tqdm(true_theta_log)

result_log = Parallel(n_jobs=num_cores)(delayed(single_inverse)(true_theta, arg, env, agent, x_traj_log[n], a_traj_log[n], filename, n) for n, true_theta in enumerate(inputs))
torch.save(result_log, '../firefly-inverse-data/data/'+filename +"EP"+str(arg.NUM_EP)+ str(np.around(arg.PI_STD, decimals = 2))+'_multiple_result.pkl')

print('done')