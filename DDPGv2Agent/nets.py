import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
class Actor(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super(self.__class__, self).__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        num_outputs = action_dim
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU(inplace=True)
        self.mu = nn.Linear(hidden_dim, num_outputs)

    def forward(self, inputs):
        """
        :param inputs: state = torch.cat([r, rel_ang, vel, ang_vel, time, vecL])
        inputs[:,1]: rel_ang
        inputs[:,3]: ang_vel
        """


        x = self.linear1(inputs)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)

        mu = torch.tanh(self.mu(x))
        return mu



class Critic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super(self.__class__, self).__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim




        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear_action = nn.Linear(action_dim, hidden_dim)
        self.relu1_action = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden_dim + hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU(inplace=True)
        self.V = nn.Linear(hidden_dim, 1)

    def forward(self, inputs, actions):


        x_input = self.linear1(inputs)
        x_input = self.relu1(x_input)
        x_action = self.linear_action(actions)
        x_action = self.relu1_action(x_action)
        x = torch.cat((x_input, x_action), dim=1)
        x = self.linear2(x)
        x = self.relu2(x)

        V = self.V(x)
        return V



    def optimalValue(self, inputs):
        batch_size, in_dim = inputs.size()
        x = inputs.unsqueeze(1).repeat(1, self.nactions, 1).view(-1, in_dim)
        a = self.actions.repeat(batch_size)
        v = self(x, a)
        vs = v.split(self.nactions)
        return torch.stack(list(map(torch.max, vs)))
