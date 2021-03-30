import math
import torch
import random
from collections import namedtuple, deque
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'action_prob'))

class MLPPolicy(nn.Module):
    def __init__(self, d_state, d_hidden, d_action):
        super(MLPPolicy, self).__init__()
        self.linear1 = nn.Linear(d_state, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_action)
        self.head = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return self.head(x.view(x.size(0),-1))

class VPGAgent():
    def __init__(self, device, lr, gamma=0.99):
        self.replay_buffer = []
        self.time_step = 0
        self.policy_net = None
        self.device = device
        self.gamma = torch.tensor([gamma], device=device)
        self.optimizer = None
        self.update_iters = 0
        self.lr = lr

    def act(self, state):
        action_probs = self.policy_net(state)
        action = torch.multinomial(action_probs, 1)
        action_prob = torch.gather(action_probs, 1, action)
        return action, action_prob


    def memorize(self, state, action, next_state, reward, action_prob):
        reward = torch.tensor([reward]).to(self.device)
        self.replay_buffer.append(Transition(state, action, next_state, reward, action_prob))
        return

    def learn(self):
        if self.optimizer is None:
            self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=self.lr)
        self.optimizer.zero_grad()
        accumulative_loss = 0
        total_return = 0
        for t, transition in enumerate(reversed(self.replay_buffer)):
            total_return *= self.gamma
            total_return += transition.reward
            loss = - torch.log(transition.action_prob) * total_return
            accumulative_loss += loss
            loss.backward(retain_graph=True)
        self.optimizer.step()
        self.replay_buffer = []
        return accumulative_loss
