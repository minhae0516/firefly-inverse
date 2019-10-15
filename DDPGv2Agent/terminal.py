# this code determines the criteria for STOP signal of monkey

import torch


def is_terminal_action(a, terminal_vel):
    """
    terminal is true if the action( which determines velocity) is lower that terminal_vel,
    which means the monkey stops.
    This approach only cares the action, does not depend on the position.
    """
    stop = (torch.norm(a) <= terminal_vel)

    if stop:
        return torch.ByteTensor([True])
    else:
        return torch.ByteTensor([False])


def is_terminal_B_action(x, a, goal_radius, terminal_vel):
    """
    x is just the mean of the belief state
    terminal is true if (px,py) of belief is in goal_radius, and
    the action( which determines velocity) is lower that terminal_vel, which means the monkey is stopping.
    """
    pos = x.view(-1)[:2] #x[:2]
    reached_target = (torch.norm(pos) <= goal_radius)
    stop = (torch.norm(a) <= terminal_vel)

    if reached_target and stop:
        return torch.ByteTensor([True])
    else:
        return torch.ByteTensor([False])


def is_terminal(x, a, goal_radius, terminal_vel):
    """
    x is just the mean of the belief state
    """
    ## stopped and reached_target are correlated
    scale = (goal_radius/terminal_vel)**2
    mu = torch.cat((x[:2], x[-2:]))
    Q = torch.diag(torch.Tensor([1, 1, scale, scale]))

    r = torch.sqrt(mu.matmul(Q).matmul(mu))
    if r <= goal_radius:
        return torch.ByteTensor([True]), True

    return torch.ByteTensor([False]), False

def is_terminal_velocity(x, a, goal_radius, terminal_vel):
    """
    x is just the mean of the belief state
    """
    pos = x.view(-1)[:2]
    vels = x.view(-1)[-2:]
    reached_target = (torch.norm(pos) <= goal_radius)

    if torch.norm(vels) <= terminal_vel:
        return torch.ByteTensor([True]), reached_target

    return torch.ByteTensor([False]), False


