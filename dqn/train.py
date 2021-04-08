import time
import torch
import gym

from itertools import count
from model import MLPPolicy, DQNAgent
from torch.utils.tensorboard import SummaryWriter

start = time.time()
writer = SummaryWriter()

# Hyper-parameters
BATCH_SIZE = 512
MEMORY_SIZE = 5000
LR = 0.001
test_interval = 1000
test_episodes = 100
TIMESTEPS = 10000
EPSILON_ENDT = 3000

env = gym.make('CartPole-v0')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = DQNAgent(d_actions=env.action_space.n, device=device, batch_size=BATCH_SIZE, memory_size=MEMORY_SIZE, lr=LR,
                 epsilon_endt=EPSILON_ENDT)
agent.policy_net = MLPPolicy(d_state=env.observation_space.shape[0], d_hidden=20,
                             d_action=env.action_space.n).to(device)

init = time.time()
print("Init time {}".format(init-start))

num_episode = 0
episode_t = 0

state = env.reset()
state = torch.from_numpy(state).unsqueeze_(0).to(device=device, dtype=torch.float)
while agent.time_step < TIMESTEPS:
    action = agent.act(state)
    next_state, reward, done, _ = env.step(action.item())
    episode_t += 1

    if not done:
        next_state = torch.from_numpy(next_state).unsqueeze_(0).to(device=device, dtype=torch.float)
    else:
        next_state = None

    agent.memorize(state, action, next_state, reward)

    state = next_state

    if len(agent.replay_buffer) > BATCH_SIZE:
        loss, epsilon = agent.learn()
        writer.add_scalar("loss/agent_step", loss, global_step=agent.time_step)
        writer.add_scalar("epsion/agent_step", epsilon, global_step=agent.time_step)

    average_test_reward = 0
    if agent.time_step % test_interval == 0:
        print("Testing at {}".format(agent.time_step))
        for i in range(test_episodes):
            test_state = env.reset()
            for t in count():
                test_state = torch.from_numpy(test_state).unsqueeze_(0).to(device=device, dtype=torch.float)
                test_action = agent.act(test_state, eval=True)
                test_next_state, _, test_done, _ = env.step(test_action.item())
                if test_done:
                    episode_reward = t+1
                    average_test_reward += episode_reward
                    break
                else:
                    test_state = test_next_state
        average_test_reward /= test_episodes
        writer.add_scalar("Test reward", average_test_reward, global_step=agent.time_step)
        env.reset()
    if average_test_reward > 195:
        print("Average reward over 100 episodes is {}".format(average_test_reward))
        break

    if done:
        num_episode += 1
        state = env.reset()
        state = torch.from_numpy(state).unsqueeze_(0).to(device=device, dtype=torch.float)
        writer.add_scalar("Train durations/episode", episode_t, global_step=num_episode)
        episode_t = 0

env.close()
writer.close()
end = time.time()
print('Time:{}'.format(end-start))
