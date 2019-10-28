
# collections of functions for inverse control

import torch

def trajectory(agent, theta, TOT_T, env, arg, gains_range, std_range, goal_radius_range):
    pro_gains, pro_noise_stds, obs_gains, obs_noise_stds,  goal_radius = torch.split(theta.view(-1), 2)

    x_traj = [] # true location
    obs_traj =[] # observation
    a_traj = [] # action
    b_traj = []
    x, _, _, _ = env.reset(gains_range, std_range, goal_radius_range, goal_radius, pro_gains, pro_noise_stds)
    #ox = agent.Bstep.observations(x)  # observation


    b, state, obs_gains, obs_noise_stds = agent.Bstep.reset(x, torch.zeros(1), pro_gains, pro_noise_stds, goal_radius, gains_range, std_range, obs_gains, obs_noise_stds)  # reset monkey's internal model
    episode = 0
    tot_t = 0

    while tot_t <= TOT_T:
        episode +=1
        t = torch.zeros(1)
        x_traj_ep = []
        obs_traj_ep = []
        a_traj_ep = []
        b_traj_ep = []

        while t < arg.EPISODE_LEN: # for a single FF

            action = agent.actor(state)

            next_x, reached_target = env(x, action.view(-1)) #track true next_x of monkey
            next_ox = agent.Bstep.observations(next_x)  # observation
            next_b, info = agent.Bstep(b, next_ox, action, env.box) # belief next state, info['stop']=terminal # reward only depends on belief
            next_state = agent.Bstep.Breshape(next_b, t, (pro_gains, pro_noise_stds, obs_gains, obs_noise_stds, goal_radius)) # state used in policy is different from belief

            # check time limit
            TimeEnd = (t+1 == arg.EPISODE_LEN) # if the monkey can't catch the firefly in EPISODE_LEN, reset the game.
            mask = torch.tensor([1 - float(TimeEnd)]) # mask = 0: episode is over

            x_traj_ep.append(x)
            obs_traj_ep.append(next_ox)
            a_traj_ep.append(action)
            b_traj_ep.append(b)

            x = next_x
            state = next_state
            b = next_b
            #ox = next_ox
            tot_t += 1.
            t += 1

            if info['stop'] or TimeEnd:  # if the monkey stops or pass the time limit, start the new firefly
                x, _, _, _ = env.reset(gains_range, std_range, goal_radius_range, goal_radius, pro_gains, pro_noise_stds)
                #ox = agent.Bstep.observations(x)  # observation
                b, state, _, _ = agent.Bstep.reset(x, torch.zeros(1), pro_gains, pro_noise_stds, goal_radius, gains_range, std_range, obs_gains, obs_noise_stds)  # reset monkey's internal model
                break
        x_traj.append(x_traj_ep)
        obs_traj.append(obs_traj_ep)
        a_traj.append(a_traj_ep)
        b_traj.append(b_traj_ep)
    return x_traj, obs_traj, a_traj, b_traj


def getLoss(agent, x_traj, obs_traj, a_traj, theta, env, gains_range, std_range):
    logPr = torch.FloatTensor([])

    pro_gains, pro_noise_stds, obs_gains, obs_noise_stds, goal_radius = torch.split(theta.view(-1), 2)

    env.pro_gains = pro_gains
    env.pro_noise_stds = pro_noise_stds
    env.goal_radius = goal_radius


    for ep, x_traj_ep in enumerate(x_traj):
        obs_traj_ep = obs_traj[ep]
        a_traj_ep = a_traj[ep]
        logPr_ep = torch.zeros(1)
        t = torch.zeros(1)
        b, state, _, _ = agent.Bstep.reset(x_traj_ep[0], t, pro_gains, pro_noise_stds, goal_radius, gains_range, std_range, obs_gains, obs_noise_stds)  # reset monkey's internal model

        for it, next_ox in enumerate(obs_traj_ep):
            action = agent.actor(state)
            logPr_ep = logPr_ep + ((a_traj_ep[it] - action) ** 2).sum() / 2 # + sign is because negative lor Pr
            next_b, info = agent.Bstep(b, next_ox, a_traj_ep[it], env.box)  # action: use real data
            next_state = agent.Bstep.Breshape(next_b, t, (pro_gains, pro_noise_stds, obs_gains, obs_noise_stds,
                                                     goal_radius))  # state used in policy is different from belief
            t += 1
            state = next_state
            b = next_b
            
        logPr = torch.cat([logPr, logPr_ep])


    #neglogPr = -1 * logPr
    return logPr.sum()


"""
def getLoss(agent, x_traj, obs_traj, a_traj, theta, env, gains_range, std_range):
    logPr = torch.FloatTensor([])

    pro_gains, pro_noise_stds, obs_gains, obs_noise_stds, goal_radius = torch.split(theta.view(-1), 2)

    env.pro_gains = pro_gains
    env.pro_noise_stds = pro_noise_stds
    env.goal_radius = goal_radius

    for ep, x_traj_ep in enumerate(x_traj):
        obs_traj_ep = obs_traj[ep]
        a_traj_ep = a_traj[ep]
        logPr_ep = torch.zeros(1)
        t = torch.zeros(1)
        x = x_traj_ep[0]
        b, state, _, _ = agent.Bstep.reset(x, t, pro_gains, pro_noise_stds, goal_radius, gains_range,
                                           std_range, obs_gains, obs_noise_stds)  # reset monkey's internal model

        for it, next_ox in enumerate(obs_traj_ep):
            action = agent.actor(state)

            next_x, reached_target = env(x, action.view(-1))  # track true next_x of monkey
            next_ox_ = agent.Bstep.observations(next_x)  # simulated observation

            logPr_ep = logPr_ep + ((a_traj_ep[it] - action) ** 2).sum() / 2 + (((next_ox - next_ox_)/100) ** 2).sum() / 2  # + sign is because negative lor Pr
            next_b, info = agent.Bstep(b, next_ox, a_traj_ep[it], env.box)  # action: use real data
            next_state = agent.Bstep.Breshape(next_b, t, (pro_gains, pro_noise_stds, obs_gains, obs_noise_stds,
                                                          goal_radius))  # state used in policy is different from belief
            t += 1
            state = next_state
            b = next_b
            x = next_x

        logPr = torch.cat([logPr, logPr_ep])

    return logPr.sum()
"""

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


    theta = torch.cat([pro_gains, pro_noise_stds, obs_gains, obs_noise_stds, goal_radius])
    return theta

def theta_range(theta, gains_range, std_range, goal_radius_range):

    theta[0].data.clamp_(gains_range[0], gains_range[1])
    theta[1].data.clamp_(gains_range[2], gains_range[3])  # [proc_gain_ang]

    theta[2].data.clamp_(std_range[0], std_range[1])  # [proc_vel_noise]
    theta[3].data.clamp_(std_range[2], std_range[3])  # [proc_ang_noise]

    theta[4].data.clamp_(gains_range[0], gains_range[1])  # [obs_gain_vel]
    theta[5].data.clamp_(gains_range[2], gains_range[3])  # [obs_gain_ang]

    theta[6].data.clamp_(std_range[0], std_range[1])  # [obs_vel_noise]
    theta[7].data.clamp_(std_range[2], std_range[3])  # [obs_ang_noise]

    theta[8].data.clamp_(goal_radius_range[0], goal_radius_range[1])


    return theta