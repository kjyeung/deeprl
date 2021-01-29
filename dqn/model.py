import math
import torch
import random
from collections import namedtuple, deque
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
EPS_START = 1
EPS_END = 0.1
class MLPPolicy(nn.Module):
    def __init__(self, d_state, d_hidden, d_action):
        super(MLPPolicy, self).__init__()
        self.linear1 = nn.Linear(d_state, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_hidden)
        self.head = nn.Linear(d_hidden, d_action)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.head(x.view(x.size(0),-1))

class DQNAgent():
    def __init__(self, d_actions, device, batch_size, replay_size, schedule_timesteps,
                  lr, epsilon_endt, gamma=0.99):
        self.replay_buffer = deque()
        self.time_step = 0
        self.epsilon = EPS_START
        self.policy_net = None
        self.target_net = None
        self.d_actions = d_actions
        self.device = device
        self.batch_size = batch_size
        self.replay_size = replay_size
        self.gamma = torch.tensor([gamma], device=device)
        self.optimizer = None
        self.update_iters = 0
        self.target_network_update_freq = 500
        self.schedule_timesteps = schedule_timesteps
        self.lr = lr
        self.epsilon_endt = epsilon_endt

    def act(self, state, eval=False):
        if eval:
            sample = 1
        else:
            self.time_step += 1
            # Update epsilon with linear schedule
            self.epsilon = EPS_END + \
                           max(0, (EPS_START - EPS_END) * (self.epsilon_endt - self.time_step ) / self.epsilon_endt)
            sample = random.random()
        if sample > self.epsilon:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.d_actions)]], device=self.device, dtype=torch.long)


    def memorize(self, state, action, next_state, reward):
        reward = torch.tensor([reward]).to(self.device)
        self.replay_buffer.append((state, action, next_state, reward))
        if len(self.replay_buffer) > self.replay_size:
            # print("Replay_buffer size is {} now. Popping old ones.".format(len(self.replay_buffer)))
            self.replay_buffer.popleft()
        return

    def learn(self):
        if self.target_net is None:
            self.target_net = self.policy_net
        if self.optimizer is None:
            self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=self.lr)

        batch = random.sample(self.replay_buffer, self.batch_size)

        state_batch = torch.cat([data[0] for data in batch]).to(torch.float)
        action_batch = torch.cat([data[1] for data in batch])
        reward_batch = torch.cat([data[3] for data in batch])

        next_state_batch = [data[2] for data in batch]
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_state_batch)),
                                      device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in next_state_batch if s is not None])
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = reward_batch + self.gamma * next_state_values

        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        # loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.update_iters += 1

        if self.update_iters % self.target_network_update_freq == 1:
            print("Update target network at {} iters".format(self.update_iters))
            self.target_net.load_state_dict(self.policy_net.state_dict())


        return loss, self.epsilon


class CNNPolicy(nn.Module):
    def __init__(self, h, w, outputs):
        super(CNNPolicy, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride +1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh *32
        self.head = nn.Linear(linear_input_size, outputs)


    def forward(self, x):
        x= F.relu(self.bn1(self.conv1(x)))
        x= F.relu(self.bn2(self.conv2(x)))
        x= F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
class ReplayMemory(object):
   def __init__(self, capacity):
       self.capacity = capacity
       self.memory = []
       self.position = 0

   def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

   def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

   def __len__(self):
        return len(self.memory)