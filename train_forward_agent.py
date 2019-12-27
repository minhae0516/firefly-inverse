# this file is for training agent by using DDPG.
"""
Three time variables
tot_t: number of times step since the code started
episode: the number of fireflies since the code started
t: number of time steps for the current firefly
"""
from DDPGv2Agent import Agent
from DDPGv2Agent.noise import *
from FireflyEnv import Model # firefly_task.py
from collections import deque
from DDPGv2Agent.rewards import return_reward
from Config import Config
import matplotlib.pyplot as plt
import pandas as pd

# read configuration parameters
arg = Config()


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
CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FloatTensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if CUDA else torch.ByteTensor
Tensor = FloatTensor

rewards = deque(maxlen=500)
hit_ratio_log = deque(maxlen=500)
time_log = deque(maxlen=500)
value_losses = deque(maxlen=500)
batch_size = arg.BATCH_SIZE
hit_log = []
avg_hit_ratio =[]
value_loss_log = []
policy_loss_log = []
rewards_log = []
rpt_tot = deque(maxlen=50) # reward per time (for fixed duration)
rpt_tot.append(0)
policy_loss, value_loss = torch.zeros(1), torch.zeros(1) # initialization
finetuning = 0 # this is the flag to indicate whether finetuning mode or not (if it is finetuning mode: reward is based on real location)
AGENT_STORE_FRQ = 1 #25


COLUMNS = ['total time', 'ep', 'std', 'time step', 'Policy NW loss', 'value NW loss','reward','avg_reward', 'goal',
           'a_vel', 'a_ang', 'true_r',
           'r', 'rel_ang', 'vel', 'ang_vel',
           'vecL1','vecL2','vecL3','vecL4','vecL5','vecL6','vecL7','vecL8','vecL9','vecL10',
           'vecL11','vecL12','vecL13','vecL14','vecL15',
           'process gain forward', 'process gain angular', 'process noise std forward', 'process noise std angular',
           'obs gain forward', 'obs gain angular', 'obs noise std forward', 'obs noise std angular', 'goal radius',
           'batch size', 'box_size', 'std_step_size', 'discount_factor', 'num_epochs']

ep_time_log = pd.DataFrame(columns=COLUMNS)

env = Model(arg) # build an environment
x, pro_gains, pro_noise_stds, goal_radius = env.reset(arg.gains_range, arg.std_range, arg.goal_radius_range)



tot_t = 0. # number of total time steps
episode = 0. # number of fireflies
int_t = 1 # variable for changing the world setting every EPISODE_LEN time steps

state_dim = env.state_dim
action_dim = env.action_dim
filename = arg.filename

#argument = {'filename': filename,
#            'argument': arg.__dict__}
torch.save(arg, '../firefly-inverse-data/data/'+filename+'_arg.pkl')

agent = Agent(state_dim, action_dim, arg,  filename, hidden_dim=128, gamma=arg.DISCOUNT_FACTOR, tau=0.001)

#"""
# if you want to use pretrained agent, load the data as below
# if not, comment it out
#agent.load('20191004-160540')
#"""


b, state, obs_gains, obs_noise_stds = agent.Bstep.reset(x, torch.zeros(1), pro_gains, pro_noise_stds, goal_radius, arg.gains_range, arg.std_range)  # reset monkey's internal model


# action space noise
std = 0.4 # this is for action space noise for exploration
noise = Noise(action_dim, mean=0., std=std)


while tot_t <= arg.TOT_T:
    episode += 1 # every episode starts a new firefly
    t = torch.zeros(1) # to track the amount of time steps to catch a firefly

    theta = (pro_gains, pro_noise_stds, obs_gains, obs_noise_stds, goal_radius)

    while t < arg.EPISODE_LEN: # for a single FF
        action = agent.select_action(state, action_noise = noise, param = None)  # with action noise

        next_x, reached_target = env(x, action.view(-1)) #track true next_x of monkey
        next_ox = agent.Bstep.observations(next_x)  # observation
        next_b, info = agent.Bstep(b, next_ox, action, env.box) # belief next state, info['stop']=terminal # reward only depends on belief
        next_state = agent.Bstep.Breshape(next_b, t, theta) # state used in policy is different from belief

        # reward
        reward = return_reward(episode, info, reached_target, next_b, env.goal_radius, arg.REWARD, finetuning)

        # check time limit
        TimeEnd = (t+1 == arg.EPISODE_LEN) # if the monkey can't catch the firefly in EPISODE_LEN, reset the game.
        mask = torch.tensor([1 - float(TimeEnd)]) # mask = 0: episode is over

        data = np.array([[tot_t, episode, std, t, policy_loss.item(), value_loss.item(), reward, np.mean(rewards),
                          reached_target.item(),action[0][0].item(), action[0][1].item(), torch.norm(x[0:2]).item(),
                          state[0][0].item(), state[0][1].item(), state[0][2].item(), state[0][3].item(),
                          state[0][5].item(), state[0][6].item(), state[0][7].item(), state[0][8].item(),
                          state[0][9].item(),
                          state[0][10].item(), state[0][11].item(), state[0][12].item(), state[0][13].item(),
                          state[0][14].item(),
                          state[0][15].item(), state[0][16].item(), state[0][17].item(), state[0][18].item(),
                          state[0][19].item(),
                          pro_gains[0].item(), pro_gains[1].item(), pro_noise_stds[0].item(), pro_noise_stds[1].item(),
                          obs_gains[0].item(), obs_gains[1].item(), obs_noise_stds[0].item(), obs_noise_stds[1].item(),
                          goal_radius.item(),
                          arg.BATCH_SIZE, arg.WORLD_SIZE, arg.STD_STEP_SIZE, arg.DISCOUNT_FACTOR, arg.NUM_EPOCHS]])

        df1 = pd.DataFrame(data, columns=COLUMNS)
        ep_time_log = ep_time_log.append(df1)

        if info['stop'] or TimeEnd:  # if the monkey stops or pass the time limit, start the new firefly
            next_x, pro_gains, pro_noise_stds, goal_radius = env.reset(arg.gains_range, arg.std_range, arg.goal_radius_range)
            next_b, next_state, obs_gains, obs_noise_stds = agent.Bstep.reset(next_x, torch.zeros(1), pro_gains, pro_noise_stds, goal_radius, arg.gains_range, arg.std_range)  # reset monkey's internal model

        agent.memory.push(state, action, 1 - mask, next_state, reward)



        if len(agent.memory) > 500:
            policy_loss, value_loss = agent.learn(batch_size=batch_size)
            policy_loss_log.append(policy_loss.data.clone().item())
            value_losses.append(value_loss.data.clone().item())
            if len(agent.memory) > 1000 and tot_t % 500 == 0:
                value_loss_log.append(np.mean(value_losses))



        if tot_t % 2500 == 0 and tot_t != 0:
            agent.save(filename, episode)

        # update variables
        x = next_x
        state = next_state
        b = next_b
        t += 1.
        tot_t += 1.
        rewards.append(reward[0].item())

        if tot_t % 100 == 0:
            # update action space exploration noise
            std -= arg.STD_STEP_SIZE  # exploration noise
            std = max(0.05, std)
            noise.reset(0., std)

        if tot_t % 500 == 0 and tot_t != 0:
            rewards_log.append(np.mean(rewards))


        if info['stop'] or TimeEnd: # if the monkey stops or pass the time limit, start the new firefly
            break



    hit = (info['stop']) * reached_target
    hit_log.append(hit)
    hit_ratio_log.append(hit)
    avg_hit_ratio.append(np.mean(hit_ratio_log))
    time_log.append(t) # time for recent 50 episodes
    policy_loss_log.append(policy_loss.item())
    value_loss_log.append(value_loss.item())











    if episode % 500 == 0 and episode != 0:
        ep_time_log.to_csv(path_or_buf='../firefly-inverse-data/data/' + filename + '_log.csv', index=False)

        print("Ep: {}, steps: {}, std: {:0.2f}, ave_reward: {:0.3f}, hit ratio: {:0.3f}".format(episode, np.mean(time_log), noise.scale,
                                                                                                np.mean(rewards), avg_hit_ratio[-1]))
        plt.figure()
        plt.plot(policy_loss_log)
        plt.savefig('../firefly-inverse-data/data/' + filename + 'policy_loss_log' + '.jpg', format='jpg')

        plt.figure()
        plt.plot(value_loss_log)
        plt.savefig('../firefly-inverse-data/data/' + filename + 'value_loss_log' + '.jpg', format='jpg')

        plt.figure()
        plt.plot(rewards_log)
        plt.savefig('../firefly-inverse-data/data/' + filename + 'av_reward_log' + '.jpg', format='jpg')

        plt.close('all')

        if CUDA:
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024, 1), 'kB')
            print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024, 1), 'kB')



