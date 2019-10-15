import torch
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
from parameter import *

def learning_curve(filename, x, xlabel, ylabel):
    plt.figure()
    plt.plot(x, '*')
    plt.title('reward over learning time')
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    path = './figures'
    #filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #filename = datetime.datetime.now().strftime("%Y%m%d")
    #os.makedirs(path, exist_ok=True)
    #file = next_path(path+'/'+filename+ylabel)
    plt.savefig(path+'/'+filename+ylabel+'.eps', format = 'eps')

def learning_curve_group(filename, y,  ylabel, xlabel):
    num = np.size(ylabel)
    x = y[xlabel]
    for id in range(num):
        y_sub = y[ylabel[id]]
        idx = str(num)+'1'+str(id+1)
        plt.subplot(int(idx))
        plt.plot(x, y_sub,'*') #, label=ylabel[id])
        #plt.legend()
        plt.title(ylabel[id])
        plt.tight_layout()
    path = './figures'
    #filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #filename = datetime.datetime.now().strftime("%Y%m%d")
    #os.makedirs(path, exist_ok=True)
    #file = next_path(path+'/'+filename+ylabel)
    #plt.savefig(path+'/'+filename+'_group'+'.eps', format = 'eps')
    plt.savefig(path+'/'+filename+'_group'+'.png', format = 'png')



def mplotter(x, xlabel, ylabel, true):
    plt.figure()
    data_size = np.size(x)
    plt.plot(x, '*', label='trajectory')
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if true is not None:
        plt.plot(true * np.ones(data_size), 'k--', label='true value')
    plt.legend()
    path = './figures'
    filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    plt.savefig(path+'/'+filename+ylabel+'.eps', format = 'eps')

def action_plotter(actions, noise_std):

    # for saving data
    filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = './figures'
    result = {'noise_std': noise_std, 'actions': actions}
    torch.save(result, path + '/' + filename + 'action.pkl')
    ## for loading data
    #torch.load(filename + 'action.pkl')
    # for plot
    plt.figure()
    for t in range(len(actions)):
        action_tr = actions[t].data.numpy()
        plt.plot(action_tr[:,1],action_tr[:,0],'*-')
    plt.ylabel('forward control input')
    plt.xlabel('angular control input')
    #plt.xlim((-1, 1))
    plt.title('exploration noise = %.2f' %(noise_std))
    plt.savefig(path + '/' + filename + 'actions'+'.eps', format = 'eps')

def action_time_plotter_csv(filename, actions, PROC_NOISE_STD, OBS_NOISE_STD):

    path = './figures'

    # for plot
    plt.figure()
    action_tr = actions[0]
    for t in range(0, len(actions)-1):
        action_tr = np.append(action_tr, actions[t], axis=0)

        if np.sum(actions[t+1]) == 0 or t == len(actions)-1:
            plt.plot(action_tr[2:, 1],'*-')
            action_tr = actions[0]

    plt.ylabel('angular control input')
    plt.xlabel('time step')
    plt.title("process noise std = {:.2f}, observation noise std = {:.2f}".format(PROC_NOISE_STD, OBS_NOISE_STD))
    plt.savefig(path + '/' + filename + 'actions_ang'+'.eps', format = 'eps')

    # for plot
    plt.figure()

    action_tr = actions[0]
    for t in range(0, len(actions)-1):
        action_tr = np.append(action_tr, actions[t], axis=0)

        if np.sum(actions[t+1]) == 0 or t == len(actions)-1:
            plt.plot(action_tr[2:, 0], '*-')
            action_tr = actions[0]

    plt.ylabel('forward control input')
    plt.xlabel('time step')
    plt.title("process noise std = {:.2f}, observation noise std = {:.2f}".format(PROC_NOISE_STD, OBS_NOISE_STD))
    plt.savefig(path + '/' + filename + 'actions_fow'+'.eps', format = 'eps')



def action_plotter_csv_sparse(filename, actions, PROC_NOISE_STD, OBS_NOISE_STD, SPARSITY=1):

    path = './figures'
    # for plot
    plt.figure()
    action_tr = actions[0]
    for t in range(0, len(actions)-1):
        action_tr = np.append(action_tr, actions[t], axis=0)
        #sample = len(action_tr)//SPARSITY
        if np.sum(actions[t+1]) == 0 or t == len(actions)-1:
            action_tr_sparse = action_tr[1::len(action_tr) // SPARSITY]
            #action_tr_sparse = np.append(action_tr_sparse, actions[t], axis=0)
            plt.plot(action_tr_sparse[2:, 1], action_tr_sparse[2:, 0],'*-')
            action_tr = actions[0]

    plt.ylabel('forward control input')
    plt.xlabel('angular control input')
    plt.xlim((-1, 1))
    plt.title("process noise std = {:.2f}, observation noise std = {:.2f}".format(PROC_NOISE_STD, OBS_NOISE_STD))
    plt.savefig(path + '/' + filename + 'actions_sparse'+'.eps', format = 'eps')

def action_plotter_csv(filename, actions, PROC_NOISE_STD, OBS_NOISE_STD):

    path = './figures'
    # for plot
    plt.figure()
    action_tr = actions[0]
    for t in range(0, len(actions)-1):
        action_tr = np.append(action_tr, actions[t], axis=0)

        if np.sum(actions[t+1]) == 0 or t == len(actions)-1:
            plt.plot(action_tr[2:, 1], action_tr[2:, 0],'*-')
            action_tr = actions[0]

    plt.ylabel('forward control input')
    plt.xlabel('angular control input')
    plt.xlim((-1,1))
    plt.title("process noise std = {:.2f}, observation noise std = {:.2f}".format(PROC_NOISE_STD, OBS_NOISE_STD))
    plt.savefig(path + '/' + filename + 'actions'+'.eps', format = 'eps')


def location_plotter(filename, ep,  b, x,  PROC_NOISE_STD, OBS_NOISE_STD):
    """
    based on firefly's perspective: (0,0) becomes firefly's location
    """
    path = './figures'
    # for plot
    plt.figure()
    b = np.reshape(b,(-1,5))
    bx = b[:,[0,1,2]]

    x = np.reshape(x, (-1, 5))
    tx = x[:, [0, 1, 2]]

    for t in range(1, int(ep[-1])+1):
        idx = np.where(ep == t)
        pxs = np.reshape(bx[idx, 0], (-1, 1))
        pys = np.reshape(bx[idx, 1], (-1, 1))
        angs = np.reshape(bx[idx, 2], (-1, 1))
        #vs = np.reshape(bx[idx, 3], (-1, 1))
        trajectory, = plt.plot(np.reshape(bx[idx, 0], (-1, 1)), np.reshape(bx[idx, 1], (-1, 1)))
        plt.quiver(pxs[0], pys[0], np.cos(angs[0]), np.sin(angs[0]), color=trajectory.get_color(),label = 'Belief')  # initial heading

        # true location
        pxs = np.reshape(tx[idx, 0], (-1, 1))
        pys = np.reshape(tx[idx, 1], (-1, 1))
        angs = np.reshape(tx[idx, 2], (-1, 1))
        # vs = np.reshape(bx[idx, 3], (-1, 1))
        plt.plot(np.reshape(tx[idx, 0], (-1, 1)), np.reshape(tx[idx, 1], (-1, 1)), linestyle='dashed',color=trajectory.get_color(),label = 'True')
        """
        idx = np.where(ep == t)
        plt.plot(np.reshape(bx[idx, 0],(-1,1)), np.reshape(bx[idx, 1],(-1,1)))
        plt.quiver(pxs[0], pys[0], np.cos(angs[0]), np.sin(angs[0]), color=trajectory.get_color())  # initial heading
        #plt.plot(bx[idx, 0], bx[idx, 1])
        """

    circle = plt.Circle((0., 0.), radius=GOAL_RADIUS, fill=False)
    ax = plt.gca()
    ax.add_patch(circle)
    #plt.plot(bx[:,0], bx[:,1], '*-')
    plt.ylabel('location: y axis')
    plt.xlabel('location: x axis')
    plt.xlim((-WORLD_SIZE, WORLD_SIZE))
    plt.ylim((-WORLD_SIZE, WORLD_SIZE))
    #plt.legend()
    plt.title("firefly is at (0,0) \n process noise std = {:.2f}, observation noise std = {:.2f}, sold: belief, dash: true".format(PROC_NOISE_STD, OBS_NOISE_STD))
    plt.savefig(path + '/' + filename + 'location'+'.eps', format = 'eps')

def location_plotter2(filename, ep,  b, x, PROC_NOISE_STD, OBS_NOISE_STD):
    """
    based on monkey's perspective: (0,0) becomes monkey's starting point
    """
    path = './figures'
    # for plot
    plt.figure()
    # belief
    b = np.reshape(b,(-1,5))
    bx = b[:,[0,1,2]]
    #true
    x = np.reshape(x, (-1, 5))
    tx = x[:, [0, 1, 2]]

    for t in range(1, int(ep[-1])+1):
        idx = np.where(ep == t)
        #belief
        ang = pi/2 - bx[idx[0][0],2]
        rm = np.array(((np.cos(ang), -np.sin(ang)), (np.sin(ang), np.cos(ang))))#rotation matrix
        loc_mat = np.concatenate((np.reshape(bx[idx, 0],(1,-1))-bx[idx[0][0],0],np.reshape(bx[idx, 1],(1,-1))-bx[idx[0][0],1]))
        rot_loc = np.matmul(rm, loc_mat)
        #px = np.matmul(rm, np.reshape(bx[idx, 0],(-1,1))-bx[idx[0][0],0])
        #py = np.matmul(rm, np.reshape(bx[idx, 1],(-1,1))-bx[idx[0][0],1])
        #print(rot_loc)
        if rot_loc[0][0] != 0:
            print("something wrong!")
        trajectory, = plt.plot(rot_loc[0,:], rot_loc[1,:],label = 'Belief')

        # true
        ang = pi / 2 - tx[idx[0][0], 2]
        rm = np.array(((np.cos(ang), -np.sin(ang)), (np.sin(ang), np.cos(ang))))  # rotation matrix
        loc_mat = np.concatenate(
            (np.reshape(tx[idx, 0], (1, -1)) - tx[idx[0][0], 0], np.reshape(tx[idx, 1], (1, -1)) - tx[idx[0][0], 1]))
        rot_loc = np.matmul(rm, loc_mat)
        # px = np.matmul(rm, np.reshape(bx[idx, 0],(-1,1))-bx[idx[0][0],0])
        # py = np.matmul(rm, np.reshape(bx[idx, 1],(-1,1))-bx[idx[0][0],1])
        # print(rot_loc)
        plt.plot(rot_loc[0, :], rot_loc[1, :], linestyle='dashed',color=trajectory.get_color(),label = 'True')

        #plt.plot(bx[idx, 0], bx[idx, 1])

    #circle = plt.Circle((0., 0.), radius=GOAL_RADIUS, fill=False)
    #ax = plt.gca()
    #ax.add_patch(circle)
    plt.quiver(0, 0, np.cos(pi / 2), np.sin(pi / 2))  # initial heading
    plt.ylabel('location: y axis')
    plt.xlabel('location: x axis')
    plt.xlim((-WORLD_SIZE, WORLD_SIZE))
    plt.ylim((0, WORLD_SIZE))
    #plt.legend()
    plt.title("monkey is at (0,0) \n process noise std = {:.2f}, observation noise std = {:.2f}, sold: belief, dash: true".format(PROC_NOISE_STD, OBS_NOISE_STD))
    plt.savefig(path + '/' + filename + 'location2'+'.eps', format = 'eps')

def location_plotter_arrow(filename, ep,  b, PROC_NOISE_STD, OBS_NOISE_STD):
    """
    based on firefly's perspective: (0,0) becomes firefly's location
    plot the angle as arrow
    """
    path = './figures'
    # for plot
    plt.figure()
    b = np.reshape(b,(-1,5))
    bx = b[:,:4]
    pxs_end = []
    pys_end = []
    angs_end = []


    for t in range(1, int(ep[-1])+1):
        idx = np.where(ep == t)
        pxs = np.reshape(bx[idx, 0],(-1,1))
        pys = np.reshape(bx[idx, 1],(-1,1))
        angs = np.reshape(bx[idx, 2],(-1,1))
        vs = np.reshape(bx[idx, 3],(-1,1)) # velocity

        pxs_end.append(pxs[-1])
        pys_end.append(pys[-1])
        angs_end.append(angs[-1])


        trajectory, = plt.plot(np.reshape(bx[idx, 0], (-1, 1)), np.reshape(bx[idx, 1], (-1, 1)))
        plt.quiver(pxs[0], pys[0], np.cos(angs[0]), np.sin(angs[0]), color=trajectory.get_color()) # initial heading
        #plt.quiver(pxs[:-1], pys[:-1], np.sign(vs[1:])*np.cos(angs[1:]), np.sign(vs[1:])*np.sin(angs[1:]), color=trajectory.get_color(), linewidths=np.absolute(vs[1:]))
        plt.quiver(pxs[:-1], pys[:-1], np.sign(vs[1:])*np.cos(angs[1:]), np.sign(vs[1:])*np.sin(angs[1:]))

    circle = plt.Circle((0., 0.), radius=GOAL_RADIUS, fill=False)
    ax = plt.gca()
    ax.add_patch(circle)
    #plt.plot(bx[:,0], bx[:,1], '*-')
    plt.ylabel('location: y axis')
    plt.xlabel('location: x axis')
    plt.xlim((-WORLD_SIZE, WORLD_SIZE))
    plt.ylim((-WORLD_SIZE, WORLD_SIZE))
    plt.title("firefly is at (0,0) \n process noise std = {:.2f}, observation noise std = {:.2f}".format(PROC_NOISE_STD, OBS_NOISE_STD))
    plt.savefig(path + '/' + filename + 'location_arrow'+'.eps', format = 'eps')


    plt.figure()
    circle = plt.Circle((0., 0.), radius=GOAL_RADIUS, fill=False)
    ax = plt.gca()
    ax.add_patch(circle)
    plt.quiver(pxs_end, pys_end, np.cos(angs_end), np.sin(angs_end), width=0.01, scale=80)
    plt.ylabel('location: y axis')
    plt.xlabel('location: x axis')
    plt.xlim((-WORLD_SIZE, WORLD_SIZE))
    plt.ylim((-WORLD_SIZE, WORLD_SIZE))
    plt.title("firefly is at (0,0) \n process noise std = {:.2f}, observation noise std = {:.2f}".format(PROC_NOISE_STD,
                                                                                                         OBS_NOISE_STD))
    plt.savefig(path + '/' + filename + 'stop_location_arrow' + '.eps', format='eps')


def timestep_radius_plotter(filename, ep, time, x, PROC_NOISE_STD, OBS_NOISE_STD):
    path = './figures'
    # for plot
    x_r = np.reshape(x,(-1, 5))
    loc = x_r[:,[0,1]]
    r = []
    time_step = []

    idx = np.where(time == 0) # 새로 에디소드 시작 하는 곳 인덱스
    for ep_idx in range(int(ep[-1])):
        r = np.append(r, np.linalg.norm(loc[idx[0][ep_idx]]))
        #r.append(ep_idx) = np.linalg.norm(loc[idx[0][ep_idx]])
        if ep_idx != ep[-1]-1:
            time_step = np.append(time_step, time[idx[0][ep_idx+1]-1])
            #time_step[ep_idx] = time[idx[0][ep_idx+1]-1]
        else:
            time_step = np.append(time_step, time[-1])
            #time_step[ep_idx] = time[-1]
    plt.figure()
    plt.plot(r, time_step, '*')
    plt.ylabel('trial length')
    plt.xlabel('initial radius')
    plt.title("pron std = {:.2f}, observation noise std = {:.2f}".format(PROC_NOISE_STD, OBS_NOISE_STD))
    plt.savefig(path + '/' + filename + 'radius+time'+'.eps', format = 'eps')
