import math
import random
import numpy as np
from itertools import count
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from model import DQN, ReplayMemory, Transition

import torch, gym
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

BATCH_SIZE = 512
GAMMA = 0.999
EPS_START = 1
EPS_END = 0.05
EPS_DECAY = 1000
TARGET_UPDATE = 4

env = gym.make('CartPole-v0').unwrapped
env.reset()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)

def get_screen():
    # Transpose screen(HWC) into torch order(CHW)
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height * 0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2, cart_location + view_width // 2)
    screen = screen[:, :, slice_range]

    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)

    return resize(screen).unsqueeze(0).to(device)

init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

n_actions = env.action_space.n

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.RMSprop(policy_net.parameters(), lr=0.001)
memory = ReplayMemory(1000000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_thershold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done/ EPS_DECAY)
    steps_done += 1
    if sample > eps_thershold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

episode_durations = []

running_loss = 0
iteration = 0
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    global running_loss
    running_loss += loss
    global iteration
    iteration += 1
    if iteration % 10 == 0:
        print("loss:{}, iter:{}".format(running_loss/10, iteration))
        running_loss = 0

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss

num_episodes = 100
for i_episode in range(num_episodes):
    env.reset()
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen
    for t in count():
        action = select_action(state)
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        last_screen = current_screen
        current_screen = get_screen()
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        memory.push(state, action, next_state, reward)
        state = next_state

        loss = optimize_model()
        if loss is not None:
          writer.add_scalar("Loss/global_step", loss, global_step=steps_done)
        if done:
            episode_durations.append(t+1)
            writer.add_scalar("Duration/episode", t+1, global_step=i_episode)
            break
    if i_episode % TARGET_UPDATE == 0:
        print("Updating target network.\n")
        target_net.load_state_dict(policy_net.state_dict())
print("Training Complete.")
env.render()
env.close()
writer.close()




