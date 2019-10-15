# this file is for training agent by using DDPG.
"""
Three time variables
tot_t: number of times step since the code started
episode: the number of fireflies since the code started
t: number of time steps for the current firefly
"""
from parameter import *  # includes all capital letter variables which are not defined in this file
from DDPGv2Agent import Agent
from DDPGv2Agent.noise import *
from FireflyEnv import Model # firefly_task.py
from collections import deque
from mplotter import *
from DDPGv2Agent.rewards import *

# fix random seed
import random
random.seed(SEED_NUMBER)
import torch
torch.manual_seed(SEED_NUMBER)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED_NUMBER)
import numpy as np
np.random.seed(SEED_NUMBER)
import datetime
import pandas as pd
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
if COS_ACTION_NOISE:
    filename = filename +'_COS'
if ACTION_NOISE and not COS_ACTION_NOISE:
    filename = filename +'_action'
if PARAM_NOISE:
    filename = filename + '_param'
if FINETUNING:
    filename = filename + '_finetune'

#filename = 'Action_COS_batch64_ProcN5e-1_ObsN5e-1'
filename = filename + '_BS' + str(BATCH_SIZE)
filename = filename + '_SNR' + str(format(SNR_dB,'.2f'))
filename = filename + '_prog_forw' + str(format(gains[0].item(),'.1f'))
filename = filename + '_prog_ang' + str(format(gains[1].item(),'.2f'))
filename = filename + '_pron_forw' + str(format(PROC_NOISE_STD[0].item(),'.3f'))
filename = filename + '_pron_ang' + str(format(PROC_NOISE_STD[1].item(),'.3f'))
filename = filename + '_obsg_forw' + str(format(obs_gains[0].item(),'.1f'))
filename = filename + '_obsg_ang' + str(format(obs_gains[1].item(),'.2f'))
filename = filename + '_obsn_forw' + str(format(OBS_NOISE_STD[0].item(),'.3f'))
filename = filename + '_obsn_ang' + str(format(OBS_NOISE_STD[1].item(),'.3f'))



#'batch size', 'process gain forward', 'process gain angular', 'process noise std forward', 'process noise std angular','obs gain forward', 'obs gain angular', 'obs noise std forward', 'obs noise std angular'
#BATCH_SIZE, gains[0].item(), gains[1].item(), PROC_NOISE_STD[0].item(), PROC_NOISE_STD[1].item(),gains[0].item(), gains[1].item(), PROC_NOISE_STD[0].item(), PROC_NOISE_STD[1].item(),obs_gains[0].item(),obs_gains[1].item(), OBS_NOISE_STD[0].item(),OBS_NOISE_STD[1].item()



rpt_ff = deque(maxlen=50) #reward per time (for a firefly)
hit_ratio_log = deque(maxlen=50)
time_log = deque(maxlen=50)
batch_size = BATCH_SIZE
hit_log = []
avg_hit_ratio =[]
value_loss_log = []
policy_loss_log = []
rpt_tot = deque(maxlen=50) # reward per time (for fixed duration)
rpt_tot.append(0)
policy_loss, value_loss = torch.zeros(1), torch.zeros(1) # initialization
finetuning = 0 # this is the flag to indicate whether finetuning mode or not (if it is finetuning mode: reward is based on real location)
AGENT_STORE_FRQ = 1 #25
#reward_log = []

# parameter space noise
if PARAM_NOISE:
    action_log = deque(maxlen=PARAM_NOISE_ADAPT_INTERVAL)# when parameter noise is used, record perturbed actor's actions
    action_noparam_log = deque(maxlen=PARAM_NOISE_ADAPT_INTERVAL) # when parameter noise is used, record pure actor's actions
    ddpg_dist = None # distance between actions
    param_noise = AdaptiveParamNoiseSpec(initial_stddev=INITIAL_STDDEV, desired_action_stddev=DESIRED_ACTION_STDDEV, adaptation_coefficient=ADAPTATION_COEFFICIENT) # this is for parameter space noise


if ACTION_NOISE:
    COLUMNS = ['ep', 'box size', 'std', 'time step', 'av_time(50)', 'Policy NW loss', 'value NW loss', 'reward',
               'FF reward per time(50)','TOT reward per time', 'goal', 'hit_ratio(50)',
               'batch size', 'process gain forward', 'process gain angular', 'process noise std forward', 'process noise std angular',
               'obs gain forward', 'obs gain angular', 'obs noise std forward', 'obs noise std angular','SNR_dB']
if PARAM_NOISE:
    COLUMNS = ['ep', 'box size', 'parameter noise std', 'action distance', 'time step', 'av_time(50)', 'Policy NW loss',
               'value NW loss', 'reward','FF reward per time(50)','TOT reward per time', 'goal', 'hit_ratio(50)',
               'batch size', 'process gain forward', 'process gain angular', 'process noise std forward', 'process noise std angular',
               'obs gain forward', 'obs gain angular', 'obs noise std forward', 'obs noise std angular','SNR_dB']

ep_time_log = pd.DataFrame(columns=COLUMNS)

env = Model(PROC_NOISE_STD, gains) # build an environment
state_dim = env.state_dim
action_dim = env.action_dim

# action space noise
if ACTION_NOISE:
    std = 0.4 # this is for action space noise for exploration
    noise = Noise(action_dim, mean=0., std=std)


# build an agent
#agent = Agent(PROC_NOISE_STD, OBS_NOISE_STD, gains, obs_gains, rew_std, state_dim, action_dim, PARAM_NOISE, filename, hidden_dim=128, tau=0.001)

agent = Agent(PROC_NOISE_STD, OBS_NOISE_STD, gains, obs_gains, rew_std, state_dim, action_dim, PARAM_NOISE, filename, hidden_dim=128, gamma=DISCOUNT_FACTOR, tau=0.001)
if PARAM_NOISE:
    agent.perturb_actor_parameters(param_noise) # reset perturb actor network

#"""
# if you want to use pretrained ddpg, load the data as below
# if not, comment it out
#oldfilename = '20190617-222836_action_BS64_SNR50.00_prog_forw10.0_prog_ang15.71_pron_forw0.032_pron_ang0.050_obsg_forw10.0_obsg_ang15.71_obsn_forw0.316_obsn_ang0.780'
#filename_ep = oldfilename+'_ep100242.0'
#fullfile='pretrained/ddpg_minhae/'+oldfilename+'/ddpg_model_'+filename_ep
#agent.load(fullfile+'.pth.tar')
#"""

env.box = WORLD_SIZE # set the box size as real world size (you can set initial box size here if you want)
tot_t = 0. # number of total time steps
episode = 0. # number of fireflies
int_t = 1 # variable for changing the world setting every EPISODE_LEN time steps
tot_rwd = 0

while tot_t <= TOT_T:
    episode += 1 # every episode starts a new firefly
    t = torch.zeros(1) # to track the amount of time steps to catch a firefly
    x = env.reset().view(1, -1) # REAL position/velocity of monkey
    #agent.Bstep.P, agent.Bstep.ox, agent.Bstep.b, agent.Bstep.state = agent.Bstep.reset(x, t) #reset monkey's internal model #do I need observation?
    agent.Bstep.P, agent.Bstep.b, agent.Bstep.state = agent.Bstep.reset(x, t)  # reset monkey's internal model
    episode_reward = 0.
    """
    if ddpg_dist is not None: # if exist
        print("EP:{}, Time Step:{}, Parameter noise std: {:0.4f}, Distance: {:0.3f}".format(episode, int(t),
                                                                                        param_noise.current_stddev,
                                                                                        ddpg_dist))
    """

    while t < EPISODE_LEN:
        if ACTION_NOISE:
            action = agent.select_action(agent.Bstep.state, action_noise = noise, param = None)  # with action noise
        if PARAM_NOISE:
            action = agent.select_action(agent.Bstep.state, action_noise = None, param = param_noise) # action with parameter noise
            action_noparam = agent.select_action(agent.Bstep.state, action_noise = None, param = None)  # action without parameter noise
            action_noparam_log.append(action_noparam.data.cpu().numpy())
            action_log.append(action.data.cpu().numpy())
             #print(action, action_noparam)


        if PARAM_NOISE:
            # adaptive parameter noise
            if (tot_t+1) % PARAM_NOISE_ADAPT_INTERVAL == 0:
                ddpg_dist = ddpg_distance_metric(action_noparam_log, action_log)
                param_noise.adapt(ddpg_dist)
                agent.perturb_actor_parameters(param_noise)  # reset perturb actor network
                #print("EP:{}, Time Step:{}, Parameter noise std: {:0.4f}, Distance: {:0.3f}".format(episode, int(t), param_noise.current_stddev, ddpg_dist))

        next_x, reached_target = env(x, action.view(-1)) #track true next_x of monkey
        agent.Bstep.ox = agent.Bstep.observations(next_x)  # observation
        agent.Bstep.b, info = agent.Bstep(agent.Bstep.b, agent.Bstep.ox, action, env.box) # belief next state, info['stop']=terminal # reward only depends on belief
        #print(next_x - agent.Bstep.b[0])
        next_state = agent.Bstep.Breshape(agent.Bstep.b, t) # state used in policy is different from belief

        # reward
        reward = return_reward(episode, info, reached_target, agent.Bstep.b, finetuning)
        #reward_log.append(reward)
        #print(reward)

        # check time limit
        TimeEnd = (t+1 == EPISODE_LEN) # if the monkey can't catch the firefly in EPISODE_LEN, reset the game.
        mask = torch.tensor([1 - float(TimeEnd)]) # mask = 0: episode is over
        done = torch.tensor(float(TimeEnd or info['stop'])).view(1,-1) # done is used for train critic network (do not give future reward)

        # train policy network
        agent.memory.push(agent.Bstep.state, action, 1-mask, next_state, reward)
        #agent.memory.push(agent.Bstep.state, action, done, next_state, reward)

        if len(agent.memory) > 500:
            policy_loss, value_loss = agent.learn(batch_size=batch_size)
            policy_loss_log.append(policy_loss)
            value_loss_log.append(value_loss)




        # update variables
        episode_reward += reward[0].item()
        x = next_x
        agent.Bstep.state = next_state
        t += 1.
        tot_t += 1.
        tot_rwd += reward[0].item() # sum of reward for fixed amount of time : 500
        if tot_t % 500 == 0:
            rpt_tot.append(tot_rwd/500)
            tot_rwd = 0

        if ACTION_NOISE and tot_t % 100 == 0:
            # update action space exploration noise
            std -= STD_STEP_SIZE  # exploration noise
            std = max(0.05, std)
            noise.reset(0., std)

        if info['stop'] or TimeEnd: # if the monkey stops or pass the time limit, start the new firefly
            break

    hit = (info['stop']) * reached_target
    hit_log.append(hit)
    hit_ratio_log.append(hit)
    avg_hit_ratio.append(np.mean(hit_ratio_log))
    rpt_ff.append(episode_reward/t) # average reward per one time step
    time_log.append(t) # time for recent 50 episodes
    avg_rew = np.mean(rpt_ff) # average reward for a recent 100 episodes(fireflies)

    #print(reward_log)

    if ACTION_NOISE:
        df1 = pd.DataFrame(np.array([[episode, env.box, std,  t.item(), np.mean(time_log), policy_loss.item(), value_loss.item(),
                                      episode_reward, np.mean(rpt_ff), rpt_tot[-1], reached_target.item(), avg_hit_ratio[-1],
                                       BATCH_SIZE, gains[0].item(), gains[1].item(),  PROC_NOISE_STD[0].item(),
                                      PROC_NOISE_STD[1].item(),obs_gains[0].item(),obs_gains[1].item(), OBS_NOISE_STD[0].item(),OBS_NOISE_STD[1].item(), SNR_dB]]), columns=COLUMNS)
    if PARAM_NOISE:
        df1 = pd.DataFrame(np.array([[episode, env.box, param_noise.current_stddev, ddpg_dist, t.item(),
                                      np.mean(time_log), policy_loss.item(), value_loss.item(), episode_reward, np.mean(rpt_ff), rpt_tot[-1],
                                      reached_target.item(), avg_hit_ratio[-1], BATCH_SIZE, gains[0].item(),
                                      gains[1].item(), PROC_NOISE_STD[0].item(), PROC_NOISE_STD[1].item(),gains[0].item(), gains[1].item(),
                                      PROC_NOISE_STD[0].item(), PROC_NOISE_STD[1].item(),SNR_dB]]), columns=COLUMNS)

    ep_time_log = ep_time_log.append(df1)

    if episode % 25 == 0 and episode != 0:
        agent.save(filename, episode)
    #if episode != 0 and len(agent.memory) > 500 and episode % AGENT_STORE_FRQ == 0:


    if episode % 100 == 0 and episode != 0:
        plt.figure()
        plt.plot(policy_loss_log)
        plt.savefig('./figures/' + filename + 'policy_loss_log' + '.eps', format='eps')

        plt.figure()
        plt.plot(value_loss_log)
        plt.savefig('./figures/' + filename + 'value_loss_log' + '.eps', format='eps')

        learning_curve(filename, rpt_ff, xlabel='episodes', ylabel='reward per time (for a firefly)')
        learning_curve_group(filename+'_curves', y = ep_time_log, ylabel=['Policy NW loss',  'value NW loss'], xlabel='ep')
        if ACTION_NOISE:
            learning_curve_group(filename, y = ep_time_log, ylabel=['hit_ratio(50)',  'std', 'FF reward per time(50)','TOT reward per time', 'av_time(50)'], xlabel='ep')
        if PARAM_NOISE:
            learning_curve_group(filename, y=ep_time_log,
                                 ylabel=['hit_ratio(50)', 'parameter noise std', 'action distance', 'FF reward per time(50)','TOT reward per time', 'av_time(50)'], xlabel='ep')
        ep_time_log.to_csv(path_or_buf='./figures/' + filename + '_log.csv', index=False)


        if ACTION_NOISE:
            print("Ep: {}, steps: {}, std: {:0.2f}, box: {:.2f}, rew: {:0.3f}, hit ratio: {:0.3f}".format(episode, int(t),
                                                                                                    noise.scale,
                                                                                                    env.box,
                                                                                                    rpt_ff[-1].item(),
                                                                                                    avg_hit_ratio[-1]))
        if PARAM_NOISE:
            print("Ep: {}, steps: {:0.2f}, Parameter noise std: {:0.4f}, Distance: {:0.3f}, rew: {:0.3f}, hit ratio: {:0.3f}".format(episode, np.mean(time_log), param_noise.current_stddev,ddpg_dist, rpt_ff[-1].item(),avg_hit_ratio[-1]))




    if FINETUNING:
        if tot_t > 3/4*TOT_T and finetuning == 0:
        #if np.mean(hit_ratio_log) > 0.9 and std <= 0.2 and finetuning == 0:
            finetuning = 1

    if COS_ACTION_NOISE:
        if std <= 0.05:
            std = 0.4
            noise.reset(0., std)


#std

"""
learning_curve(filename, rpt_ff, xlabel='episodes', ylabel='rewards')
learning_curve_group(filename, y = ep_time_log, ylabel=['hit_ratio(50)',  'std', 'reward per time(50)', 'av_time(50)'], xlabel='ep')
ep_time_log.to_csv(path_or_buf='./figures/' + filename + '_log.csv', index=False)
print("Ep: {}, steps: {}, std: {:0.2f}, box: {:.2f}, rew: {:0.3f}, hit ratio: {:0.3f}".format(episode, int(t),
                                                                                                    noise.scale,
                                                                                                    env.box,
                                                                                                    rpt_ff[-1].item(),
                                                                                                    avg_hit_ratio[-1]))
"""
