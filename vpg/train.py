import time
import torch, gym

from itertools import count
from model import MLPPolicy, VPGAgent
from torch.utils.tensorboard import SummaryWriter

start = time.time()
writer = SummaryWriter()

NUM_EPISODES = 3000
LR =  0.001
test_interval = 100
test_episodes = 100

env = gym.make('CartPole-v0')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d_states = env.observation_space.shape[0]
d_actions = env.action_space.n

agent = VPGAgent(device, LR)
agent.policy_net = MLPPolicy(d_states, 20, d_actions).to(device)


init = time.time()
print("Init time {}".format(init-start))

for i_episode in range(NUM_EPISODES):
    state = env.reset()
    state = torch.from_numpy(state).unsqueeze_(0).to(torch.float).to(device)
    for t in count():
        action , action_prob = agent.act(state)
        next_state, reward, done, _ = env.step(action.item())

        if not done:
            next_state = torch.from_numpy(next_state).unsqueeze_(0).to(torch.float).to(device)
        else:
            next_state = None

        agent.memorize(reward, action_prob)

        state = next_state
        if done:
            writer.add_scalar("Train durations/episode", t, global_step=i_episode)
            break

    episode_loss = agent.learn()
    writer.add_scalar("Loss/episode", episode_loss, global_step=i_episode)

    if i_episode % test_interval == 0:
        print("Testing at {} episodes.".format(i_episode))
        total_test_reward = 0
        for i in range(test_episodes):
            test_state = env.reset()
            for t in count():
                test_state = torch.from_numpy(test_state).unsqueeze_(0).to(device).to(torch.float)
                test_action, _ = agent.act(test_state)
                test_next_state, _, test_done, _ = env.step(test_action.item())
                if test_done:
                    episode_reward = t+1
                    total_test_reward += episode_reward
                    break
                else:
                    test_state = test_next_state
        average_test_reward = total_test_reward/test_episodes
        writer.add_scalar("Test reward", average_test_reward, global_step=i_episode)
        if average_test_reward > 195:
            print("Average reward over 100 episodes is {}".format(average_test_reward))
            break


env.close()
writer.close()
end = time.time()
print('Time:{}'.format(end-start))
#
#
#
#
