import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

import game

learning_rate = 0.01
gamma = 0.99
max_time = 1000


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.state_space = game.state_size
        self.action_space = game.action_size

        self.l1 = nn.Linear(self.state_space, 360, bias=False)
        self.l2 = nn.Linear(360, self.action_space, bias=False)

        self.gamma = gamma

        # Episode policy and reward history
        self.policy_history = Variable(torch.Tensor())
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []

    def forward(self, x):
        model = torch.nn.Sequential(
            self.l1,
            nn.Dropout(p=0.6),
            nn.ReLU(),
            self.l2,
            nn.Softmax(dim=-1)
        )
        return model(x)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

f = True


def select_action(state):
    global f

    # Select an action (0 or 1) by running policy model and choosing based on the probabilities in state
    state = torch.from_numpy(state).type(torch.FloatTensor)
    state = policy(Variable(state))
    c = Categorical(state)
    action = c.sample()

    # Add log probability of our chosen action to our history
    if not f:
        policy.policy_history = torch.cat([policy.policy_history, c.log_prob(action).view(1)])
    else:
        policy.policy_history = c.log_prob(action).view(1)
    return action


def update_policy():
    R = 0
    rewards = []

    # Discount future rewards back to the present using gamma
    for r in policy.reward_episode[::-1]:
        R = r + policy.gamma * R
        rewards.insert(0, R)

    # Scale rewards
    rewards = torch.FloatTensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

    # Calculate loss
    loss = (torch.sum(torch.mul(policy.policy_history, Variable(rewards)).mul(-1), -1))

    # Update network weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Save and intialize episode history counters
    policy.loss_history.append(loss.item())
    policy.reward_history.append(np.sum(policy.reward_episode))
    policy.policy_history = Variable(torch.Tensor())
    policy.reward_episode = []


def train(episodes, save=0):
    global f

    running_reward = 1
    for episode in range(episodes):
        state = game.reset()  # Reset environment and record the starting state
        f = True

        game_reward = 0

        for time in range(max_time):
            action = select_action(state)
            f = False
            # Step through environment using chosen action
            state, reward, done = game.step(action.item())

            if action.item() == 1:
                x = 0
            if action.item() == 2:
                x = 0

            # Save reward
            policy.reward_episode.append(reward)
            game_reward += reward
            if done:
                break

        # Used to determine when the environment is solved.
        running_reward = (running_reward * 0.99) + (game_reward * 0.01)

        update_policy()

        if episode % 50 == 0:
            print('Episode {}\tLast reward: {:5d}\tAverage reward: {:.2f}'.format(episode, game_reward, running_reward))

        if save != 0 and (episode + 1) % save == 0:
            PATH = 'models/tetris_policy_' + str(episode + 1) + '.pth'
            torch.save(policy.state_dict(), PATH)


if __name__== "__main__":
    train(episodes=10000, save=500)
