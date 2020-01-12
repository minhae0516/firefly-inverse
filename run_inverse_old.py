import torch
import torch.nn as nn
from torch.autograd import grad
import pandas as pd
from InverseFuncs import trajectory, getLoss, reset_theta, theta_range


from DDPGv2Agent import Agent
from FireflyEnv import Model # firefly_task.py
from collections import deque
from Inverse_Config import Inverse_Config
#import matplotlib.pyplot as plt

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


# #filename = '20191014-180128' #studpid agent
# filename = '20191015-161114-2'
# df = pd.read_csv('../firefly-inverse-data/data/' + filename + '_log.csv', usecols=['discount_factor'])
# DISCOUNT_FACTOR = df['discount_factor'][0]


filename = '20191016-205855-5' # agent information
df = pd.read_csv('../firefly-inverse-data/data/' + filename + '_log.csv',
                 usecols=['discount_factor','process gain forward', 'process gain angular', 'process noise std forward',
                          'process noise std angular', 'obs gain forward', 'obs gain angular', 'obs noise std forward',
                          'obs noise std angular', 'goal radius'])

DISCOUNT_FACTOR = df['discount_factor'][0]
arg.gains_range = [np.floor(df['process gain forward'].min()), np.ceil(df['process gain forward'].max()),
               np.floor(df['process gain angular'].min()), np.ceil(df['process gain angular'].max())]

arg.std_range = [df['process noise std forward'].min(), df['process noise std forward'].max(),
               df['process noise std angular'].min(), df['process noise std angular'].max()]
arg.goal_radius_range = [df['goal radius'].min(), df['goal radius'].max()]


env = Model(arg) # build an environment
agent = Agent(env.state_dim, env.action_dim, arg,  filename, hidden_dim=128, gamma=DISCOUNT_FACTOR, tau=0.001) #, device = "cpu")
agent.load(filename)

true_theta = reset_theta(arg.gains_range, arg.std_range, arg.goal_radius_range)
x_traj, obs_traj, a_traj, _ = trajectory(agent, true_theta, arg.INVERSE_BATCH_SIZE, env, arg, arg.gains_range, arg.std_range, arg.goal_radius_range) # generate true trajectory
true_loss = getLoss(agent, x_traj, obs_traj, a_traj, true_theta, env, arg.gains_range, arg.std_range) # this is the lower bound of loss?


#theta = nn.Parameter(true_theta.data.clone()+0.5*true_theta.data.clone())
theta = nn.Parameter(reset_theta(arg.gains_range, arg.std_range, arg.goal_radius_range))
ini_theta = theta.data.clone()


loss_log = deque(maxlen=1000)
theta_log = deque(maxlen=1000)
optT = torch.optim.Adam([theta], lr=1e-3)
prev_loss = 100000
loss_diff = deque(maxlen=5)


for num_batches in range(10000):
    loss = getLoss(agent, x_traj, obs_traj, a_traj, theta, env, arg.gains_range, arg.std_range)
    loss_log.append(loss.data)
    optT.zero_grad()
    loss.backward(retain_graph=True)
    optT.step() # performing single optimize step: this changes theta
    theta = theta_range(theta, arg.gains_range, arg.std_range, arg.goal_radius_range) # keep inside of trained range
    theta_log.append(theta.data.clone())
    if loss < true_loss:
        print('loss:', loss.data, 'true loss:', true_loss.data)

    #if torch.abs(prev_loss - loss) < 1e-4:
     #   break
    loss_diff.append(torch.abs(prev_loss - loss))

    if num_batches > 5 and np.sum(loss_diff) < 1e-3:
        break
    prev_loss = loss.data

    if num_batches%100 == 0:
        print("num:{}, loss:{}, true_loss:{}".format(num_batches, np.round(loss.data.item(), 6), true_loss.data))
        print("num:{},theta diff sum:{}".format(num_batches, 1e6 * (true_theta - theta.data.clone()).sum().data))
        print("num:{}, initial_theta:{}, \n converged_theta:{},\n true theta:{}".format(num_batches, ini_theta, theta.data.clone(),
                                                                                true_theta))

#
loss = getLoss(agent, x_traj, obs_traj, a_traj, theta, env, arg.gains_range, arg.std_range)
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