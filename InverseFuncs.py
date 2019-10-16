
# collections of functions for inverse control

import torch

def trajectory(agent, theta, TOT_T, env, arg, gains_range, std_range, goal_radius_range):
    pro_gains, pro_noise_stds, obs_gains, obs_noise_stds,  goal_radius = torch.split(theta.view(-1), 2)

    x_traj = [] # true location
    obs_traj =[] # observation
    a_traj = [] # action
    x, _, _, _ = env.reset(gains_range, std_range, goal_radius_range, goal_radius, pro_gains, pro_noise_stds)

    #agent = Agent(env.state_dim, env.action_dim, arg,  filename, hidden_dim=128, gamma=DISCOUNT_FACTOR, tau=0.001)
    #agent.load(filename)

    b, state, obs_gains, obs_noise_stds = agent.Bstep.reset(x, torch.zeros(1), pro_gains, pro_noise_stds, goal_radius, gains_range, std_range, obs_gains, obs_noise_stds)  # reset monkey's internal model
    episode = 0
    tot_t = 0

    while tot_t <= TOT_T:
        episode +=1
        t = torch.zeros(1)

        while t < arg.EPISODE_LEN: # for a single FF

            #state = Variable(state).to(device)
            #action = agent.actor(state).detach()
            action = agent.actor(state)

            #action = agent.select_action(state)

            next_x, reached_target = env(x, action.view(-1)) #track true next_x of monkey
            ox = agent.Bstep.observations(next_x)  # observation
            next_b, info = agent.Bstep(b, ox, action, env.box) # belief next state, info['stop']=terminal # reward only depends on belief
            next_state = agent.Bstep.Breshape(next_b, t, (pro_gains, pro_noise_stds, obs_gains, obs_noise_stds, goal_radius)) # state used in policy is different from belief

            # check time limit
            TimeEnd = (t+1 == arg.EPISODE_LEN) # if the monkey can't catch the firefly in EPISODE_LEN, reset the game.
            mask = torch.tensor([1 - float(TimeEnd)]) # mask = 0: episode is over

            x_traj.append(x)
            obs_traj.append(ox)
            a_traj.append(action)

            x = next_x
            state = next_state
            tot_t += 1.
            t += 1

            if info['stop'] or TimeEnd:  # if the monkey stops or pass the time limit, start the new firefly
                x, _, _, _ = env.reset(gains_range, std_range, goal_radius_range, goal_radius, pro_gains, pro_noise_stds)
                b, state, _, _ = agent.Bstep.reset(next_x, torch.zeros(1), pro_gains, pro_noise_stds, goal_radius, gains_range, std_range, obs_gains, obs_noise_stds)  # reset monkey's internal model
                break
    return x_traj, obs_traj, a_traj


def getLoss(agent, x_traj, obs_traj, a_traj, filename, theta, env, arg, DISCOUNT_FACTOR, gains_range, std_range):
    logPr = 0#torch.FloatTensor([0])

    pro_gains, pro_noise_stds, obs_gains, obs_noise_stds, goal_radius = torch.split(theta.view(-1), 2)

    env.pro_gains = pro_gains
    env.pro_noise_stds = pro_noise_stds
    env.goal_radius = goal_radius

    #agent = Agent(env.state_dim, env.action_dim, arg, filename, hidden_dim=128, gamma=DISCOUNT_FACTOR, tau=0.001)
    #agent.load(filename)
    t = torch.zeros(1)
    b, state, _, _ = agent.Bstep.reset(x_traj[0], t, pro_gains, pro_noise_stds, goal_radius, gains_range, std_range, obs_gains, obs_noise_stds)  # reset monkey's internal model


    for it, ox in enumerate(obs_traj):
        #action = agent.select_action(state, action_noise=None, param=None)  # with action noise
        action = agent.actor(state)
        logPr = logPr - ((a_traj[it] - action) ** 2).sum() / 2

        next_b, info = agent.Bstep(b, ox, a_traj[it], env.box)  # belief next state, info['stop']=terminal # reward only depends on belief
        if info['stop']:
            # use x only the the beginning of a FF
            t = torch.zeros(1)
            b, state, _, _ = agent.Bstep.reset(x_traj[it], t, pro_gains, pro_noise_stds, goal_radius, gains_range, std_range, obs_gains, obs_noise_stds)  # reset monkey's internal model

        else:
            state = agent.Bstep.Breshape(next_b, t, (pro_gains, pro_noise_stds, obs_gains, obs_noise_stds, goal_radius))  # state used in policy is different from belief
            t += 1

    neglogPr = -logPr
    return neglogPr

def reset_theta(gains_range, std_range, goal_radius_range):
    pro_gains = torch.zeros(2)
    pro_noise_stds = torch.zeros(2)
    obs_gains = torch.zeros(2)
    obs_noise_stds = torch.zeros(2)

    pro_gains[0] = torch.zeros(1).uniform_(gains_range[0], gains_range[1])  # [proc_gain_vel]
    pro_gains[1] = torch.zeros(1).uniform_(gains_range[2], gains_range[3])  # [proc_gain_ang]

    pro_noise_stds[0] = torch.zeros(1).uniform_(std_range[0], std_range[1])  # [proc_vel_noise]
    pro_noise_stds[1] = torch.zeros(1).uniform_(std_range[2], std_range[3])  # [proc_ang_noise]

    obs_gains[0] = torch.zeros(1).uniform_(gains_range[0], gains_range[1])  # [obs_gain_vel]
    obs_gains[1] = torch.zeros(1).uniform_(gains_range[2], gains_range[3])  # [obs_gain_ang]

    obs_noise_stds[0] = torch.zeros(1).uniform_(std_range[0], std_range[1])  # [obs_vel_noise]
    obs_noise_stds[1] = torch.zeros(1).uniform_(std_range[2], std_range[3])  # [obs_ang_noise]

    goal_radius = torch.zeros(1).uniform_(goal_radius_range[0], goal_radius_range[1])



    #rew_std = GOAL_RADIUS / 2 / 2 * torch.zeros(1).uniform_(rew_std_range[0], rew_std_range[1])  # 2*std of Gaussian distribution for reward: 95%



    theta = torch.cat([pro_gains, pro_noise_stds, obs_gains, obs_noise_stds, goal_radius])
    return theta
