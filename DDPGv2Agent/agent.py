
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam # use Adam optimizer for deep neural net

from .nets import Actor, Critic
#from .nets_norm import Actor, Critic

from .utils import *
from .ReplayMemory import ReplayMemory
from .belief_step import BeliefStep # belief update of the agent

#CUDA = torch.cuda.is_available()
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent():
    def __init__(self, input_dim, action_dim, arg, filename=None, hidden_dim=128, gamma=0.99, tau=0.001, memory_size=1e6, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):

        self.device = device
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.tau = tau
        print("Running DDPG Agent: using ", self.device)

        self.actor = Actor(input_dim, action_dim, hidden_dim).to(self.device)
        self._actor = Actor(input_dim, action_dim, hidden_dim).to(self.device)  # target NW
        self.critic = Critic(input_dim, action_dim, hidden_dim).to(self.device)
        self._critic = Critic(input_dim, action_dim, hidden_dim).to(self.device)# target NW


        self.actor_optim = Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optim = Adam(self.critic.parameters(), lr=1e-3)

        self.priority = False
        self.memory = ReplayMemory(int(memory_size), priority=self.priority)

        self.args = (input_dim, action_dim, hidden_dim)
        hard_update(self._actor, self.actor)  # Make sure target is with the same weight
        hard_update(self._critic, self.critic)
        self.create_save_file(filename)


        # for belief step
        self.Bstep = BeliefStep(arg)




    def select_action(self,  state, action_noise=None, param = None):

        state = Variable(state).to(self.device)

        if param is not None:
            mu = self.actor_perturbed(state).detach()
        else: # no parameter space noise
            mu = self.actor(state).detach()

        if action_noise is not None:
            mu += torch.Tensor(action_noise.noise()).to(self.device)
        return mu.clamp(-1, 1)

    def update_parameters(self, batch):
        states = variable(torch.cat(batch.state))
        next_states = variable(torch.cat(batch.next_state))
        actions = variable(torch.cat(batch.action))
        rewards = variable(torch.cat(batch.reward).unsqueeze(1))
        #masks = variable(torch.cat(batch.mask))
        dones = variable(torch.cat(batch.done))

        with torch.no_grad():
            next_actions = self._actor(next_states) # use target
            next_qvalues = self._critic(next_states, next_actions) # use target network
            #next_qvalues = self._critic(next_states, next_actions) * (1 - dones)
            target_qvalues = rewards + self.gamma * next_qvalues

        self.critic_optim.zero_grad()
        pred_qvalues = self.critic(states, actions)
        value_loss = torch.mean((pred_qvalues - target_qvalues)**2)
        value_loss.backward()
        self.critic_optim.step()

        self.actor_optim.zero_grad()
        policy_loss = -self.critic(states, self.actor(states))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()
        return policy_loss, value_loss

    def learn(self, epochs=2, batch_size=64):
        for epoch in range(epochs):
            # sample new batch here
            batch, _ = self.memory.sample(batch_size)
            losses = self.update_parameters(batch)
            soft_update(self._actor, self.actor, self.tau)
            soft_update(self._critic, self.critic, self.tau)

        return losses

    def save(self, filename, episode):
        state = {
            'args': self.args,
            'actor_dict': self.actor.state_dict(),
            'critic_dict': self.critic.state_dict(),
        }

        #self.file = '../firefly-inverse-data/trained_agent/' + filename + '.pth.tar'

        # if you want a single file, comment out this line
        #path = './pretrained/ddpg_minhae/'+filename
        #os.makedirs(path, exist_ok=True)
        #self.file = path + '/' + 'ddpg_model_' + filename + '_ep'+ str(episode)+'.pth.tar'

        torch.save(state, self.file)
        if episode % 100 == 0:
            print("Saved to " + self.file)

    def load(self, filename):
        file = '../firefly-inverse-data/trained_agent/'+filename+'.pth.tar'
        state = torch.load(file, map_location=lambda storage, loc: storage)
        if self.args != state['args']:
            print('Agent parameters from file are different from call')
            print('Overwriting agent to load file ... ')
            args = state['args']
            #self = Agent(*args)
            self.__init__(*args)

        self.actor.load_state_dict(state['actor_dict'])
        self.critic.load_state_dict(state['critic_dict'])
        hard_update(self._actor, self.actor)  # Make sure target is with the same weight
        hard_update(self._critic, self.critic)
        #print('Loaded')
        return

    def create_save_file(self, filename):
        path = '../firefly-inverse-data/trained_agent'
        os.makedirs(path, exist_ok=True)
        if filename == None:
            self.file = next_path(path + '/' + 'ddpgmodel_%s.pth.tar')
        else: self.file = path + '/' + filename + '.pth.tar'

    def perturb_actor_parameters(self, param_noise):
        """Apply parameter noise to actor model, for exploration"""
        hard_update(self.actor_perturbed, self.actor)
        params = self.actor_perturbed.state_dict()
        for name in params:
            if 'ln' in name:
                pass
            param = params[name]
            if 'bn' not in name:
                random = torch.randn(param.shape).to(self.device)
                
                param += random * param_noise.current_stddev