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
x_traj, obs_traj, a_traj, _ = trajectory(agent, true_theta, env, arg, arg.gains_range, arg.std_range, arg.goal_radius_range, arg.NUM_EP) # generate true trajectory
true_loss = getLoss(agent, x_traj,  a_traj, true_theta, env, arg.gains_range, arg.std_range, arg.PI_STD, arg.NUM_SAMPLES) # this is the lower bound of loss?
print("true loss:{}".format(true_loss))
print("true_theta:{}".format(true_theta))






#theta = nn.Parameter(true_theta.data.clone()+0.1*true_theta.data.clone())
theta = nn.Parameter(reset_theta(arg.gains_range, arg.std_range, arg.goal_radius_range))
ini_theta = theta.data.clone()


loss_log = deque(maxlen=10000)
theta_log = deque(maxlen=10000)
optT = torch.optim.Adam([theta], lr=arg.ADAM_LR)
prev_loss = 100000
loss_diff = deque(maxlen=5)


for num_batches in range(10000):
    loss = getLoss(agent, x_traj, a_traj, theta, env, arg.gains_range, arg.std_range, arg.PI_STD, arg.NUM_SAMPLES)
    loss_log.append(loss.data)
    optT.zero_grad()
    loss.backward(retain_graph=True)
    optT.step() # performing single optimize step: this changes theta
    theta = theta_range(theta, arg.gains_range, arg.std_range, arg.goal_radius_range) # keep inside of trained range
    theta_log.append(theta.data.clone())

    #if loss < true_loss:
    #    print('loss:', loss.data, 'true loss:', true_loss.data)

    #if torch.abs(prev_loss - loss) < 1e-4:
     #   break
    loss_diff.append(torch.abs(prev_loss - loss))

    if num_batches > 5 and np.sum(loss_diff) < 1e-2:
        break
    prev_loss = loss.data

    if num_batches%100 == 0:
        print("num:{}, loss:{}".format(num_batches, np.round(loss.data.item(), 6)))
        #print("num:{},theta diff sum:{}".format(num_batches, 1e6 * (true_theta - theta.data.clone()).sum().data))
        print("num:{}, initial_theta:{}, \n converged_theta:{}".format(num_batches, ini_theta, theta.data.clone()))

#
loss = getLoss(agent, x_traj, a_traj, theta, env, arg.gains_range, arg.std_range, arg.PI_STD, arg.NUM_SAMPLES)
print("loss:{}".format(loss))

toc = time.time()
print((toc - tic)/60/60, "hours")


grads = grad(loss, theta, create_graph=True)[0]
H = torch.zeros(9,9)
for i in range(9):
    H[i] = grad(grads[i], theta, retain_graph=True)[0]
I = H.inverse()
stderr = torch.sqrt(I.diag())


result = {'true_theta': true_theta,
          'initial_theta': ini_theta,
          'theta': theta,
          'theta_log': theta_log,
          'loss_log': loss_log,
          'filename': filename,
          'num_batches': num_batches,
          'duration': toc-tic,
          'arguments': arg,
          'stderr': stderr
          }
torch.save(result, '../firefly-inverse-data/data/'+filename+'_result.pkl')

print('done')