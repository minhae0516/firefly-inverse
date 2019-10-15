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
        self.ln1 = nn.LayerNorm(hidden_dim)  # layer normalization
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)  # layer normalization
        self.relu2 = nn.ReLU(inplace=True)
        self.mu = nn.Linear(hidden_dim, num_outputs)
        self.tanh = nn.Tanh()

    def forward(self, inputs):

        x = self.linear1(inputs)
        x = self.ln1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.ln2(x)
        x = self.relu2(x)
        x = self.mu(x)
        mu = self.tanh(x)
        return mu

class Actor_(nn.Module): # target network
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super(self.__class__, self).__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        num_outputs = action_dim

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # batch normalization
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(hidden_dim)  # batch normalization
        self.mu = nn.Linear(hidden_dim, num_outputs)
        self.tanh = nn.Tanh()

    def forward(self, inputs):
        x = self.linear1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.linear2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.mu(x)
        mu = self.tanh(x)

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
        self.relu_action = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden_dim + hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU(inplace=True)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.relu3 = nn.ReLU(inplace=True)
        self.Q = nn.Linear(hidden_dim, 1)

    def forward(self, inputs, actions):
        x = self.linear1(inputs)
        x = self.relu1(x)
        a = self.linear_action(actions)
        a = self.relu_action(a)
        x = torch.cat((x, a), dim=1)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)

        Q = self.Q(x)
        return Q

class Critic_(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super(self.__class__, self).__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear_action = nn.Linear(action_dim, hidden_dim)
        self.relu_action = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden_dim + hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU(inplace=True)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.relu3 = nn.ReLU(inplace=True)
        self.Q = nn.Linear(hidden_dim, 1)

        """
        x = np.arange(-1, 1.005, 0.05)
        x,y = np.meshgrid(x, x)
        grid = np.vstack([x.ravel(), y.ravel()])
        self.actions = torch.Tensor(grid).t()
        self.nactions = self.actions.size(0)
        """

    def forward(self, inputs, actions):
        x = self.linear1(inputs)
        x = self.relu1(x)
        a = self.linear_action(actions)
        a = self.relu_action(a)
        x = torch.cat((x, a), dim=1)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)

        Q = self.Q(x)
        return Q

    def optimalValue(self, inputs):
        batch_size, in_dim = inputs.size()
        x = inputs.unsqueeze(1).repeat(1, self.nactions, 1).view(-1, in_dim)
        a = self.actions.repeat(batch_size)
        v = self(x, a)
        vs = v.split(self.nactions)
        return torch.stack(list(map(torch.max, vs)))
