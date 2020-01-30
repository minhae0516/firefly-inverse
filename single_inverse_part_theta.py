import torch
from tqdm import tqdm

import torch.nn as nn
from torch.autograd import grad
from InverseFuncs import getLoss, reset_theta, theta_range

from collections import deque

import torch
import numpy as np
import time



def single_inverse(true_theta, arg, env, agent, x_traj, a_traj, filename, n, Pro_Noise = None, Obs_Noise = None):
    tic = time.time()

    if Pro_Noise is not None:
        Pro_Noise = true_theta[2:4]
    if Obs_Noise is not None:
        Obs_Noise = true_theta[6:8]

    rndsgn = torch.sign(torch.randn(1, len(true_theta))).view(-1)
    purt = torch.Tensor([0.5, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0.1])  # perturbation

    theta = nn.Parameter(true_theta.data.clone() + rndsgn * purt)
    theta = theta_range(theta, arg.gains_range, arg.std_range, arg.goal_radius_range)  # keep inside of trained range


    #theta = nn.Parameter(reset_theta(arg.gains_range, arg.std_range, arg.goal_radius_range, Pro_Noise, Obs_Noise))
    ini_theta = theta.data.clone()
    print("num_theta:{}, lr:{} loss:{}\n initial theta:{}\n".format(n, scheduler.get_lr(),
                                                                              np.round(loss.data.item(), 6),
                                                                              theta.data.clone()))

    loss_log = deque(maxlen=arg.NUM_IT)
    loss_act_log = deque(maxlen=arg.NUM_IT)
    loss_obs_log = deque(maxlen=arg.NUM_IT)
    theta_log = deque(maxlen=arg.NUM_IT)
    optT = torch.optim.Adam([theta], lr=arg.ADAM_LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optT, step_size=arg.LR_STEP,
                                                gamma=0.95)  # decreasing learning rate x0.5 every 100steps


    for it in tqdm(range(arg.NUM_IT)):

        """
        if it % 100 == 0:
            print("num:{}, it:{}/{}\n".format(n, it, arg.NUM_IT))
        """

        loss, loss_act, loss_obs = getLoss(agent, x_traj, a_traj, theta, env, arg.gains_range, arg.std_range, arg.PI_STD, arg.NUM_SAMPLES)
        loss_log.append(loss.data)
        loss_act_log.append(loss_act.data)
        loss_obs_log.append(loss_obs.data)
        optT.zero_grad()
        loss.backward(retain_graph=True)
        optT.step() # performing single optimize step: this changes theta
        theta = theta_range(theta, arg.gains_range, arg.std_range, arg.goal_radius_range, Pro_Noise, Obs_Noise) # keep inside of trained range
        theta_log.append(theta.data.clone())
        if it < 50:
            scheduler.step()



        if it%5 == 0:
            print("num_theta:{}, num:{}, lr:{} loss:{}\n converged_theta:{}\n".format(n, it, scheduler.get_lr(),np.round(loss.data.item(), 6),theta.data.clone()))






    loss, _, _ = getLoss(agent, x_traj, a_traj, theta, env, arg.gains_range, arg.std_range, arg.PI_STD, arg.NUM_SAMPLES)
    #print("loss:{}".format(loss))

    toc = time.time()
    #print((toc - tic)/60/60, "hours")


    grads = grad(loss, theta, create_graph=True)[0]
    H = torch.zeros(9,9)
    for i in range(9):
        H[i] = grad(grads[i], theta, retain_graph=True)[0]
    I = H.inverse()
    stderr = torch.sqrt(I.diag())


    stderr_ii = 1/torch.sqrt(torch.abs(H.diag()))




    result = {'true_theta': true_theta,
              'initial_theta': ini_theta,
              'theta': theta,
              'theta_log': theta_log,
              'loss_log': loss_log,
              'loss_act_log': loss_act_log,
              'loss_obs_log': loss_obs_log,
              'filename': filename,
              'num_theta': n,
              'converging_it': it,
              'duration': toc-tic,
              'arguments': arg,
              'stderr': stderr,
              'stderr_ii': stderr_ii
              }


    return result