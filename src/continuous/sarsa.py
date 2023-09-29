import random
import os
from collections import namedtuple, deque
from itertools import count
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from continuous.env import ReacherEnv
from utils import sample_space, timed

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

class SARSA(nn.Module):

    # !!!!!!! changed from n_actions to 1

    def __init__(self, n_observations):
        super(SARSA, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        # self.layer2 = nn.Linear(128, 256)
        # self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, 1)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        # x = F.relu(self.layer2(x))
        # x = F.relu(self.layer3(x))
        return self.layer4(x)

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

def select_action():
    #! Check dimensions
    return torch.tensor(sample_space(ReacherEnv.act_space), device=device, dtype=torch.long)

def optimize_model(value_net, optimizer, memory, env):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    # action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_values = value_net(state_batch) #.gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        
        next_state_values[non_final_mask] = value_net(non_final_next_states).squeeze()
    # Compute the expected Q values
    expected_state_values = (next_state_values * env.discount) + reward_batch.squeeze()

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_values.squeeze(), expected_state_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(value_net.parameters(), 100)
    optimizer.step()

@timed
def train_sarsa_model(env, class_name: str, n_episodes: int = 10000):
    value_net = SARSA(len(ReacherEnv.state_space)).to(device)

    # TODO: load any models with more episodes than n_episodes; when saving a model only keep the biggest run
    file_path = f"sarsa_models/{class_name}/{n_episodes}.pt"
    if os.path.exists(file_path):
        value_net.load_state_dict(torch.load(file_path))
        return value_net
    
    optimizer = optim.AdamW(value_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)

    steps_done = 0

    for i_episode in range(n_episodes):
        # Initialize the environment and get it's state
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            action = select_action()
            obs, reward, done, info = env.step(action)
            reward = torch.tensor(np.array([reward]), device=device)

            if done:
                next_state = None
            else:
                next_state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model(value_net, optimizer, memory, env)

            if done or t > 100:
                break

    # save the model
    if not os.path.exists('sarsa_models'):
        os.makedirs('sarsa_models')
    if not os.path.exists(f'sarsa_models/{class_name}'):
        os.makedirs(f'sarsa_models/{class_name}')
    torch.save(value_net.state_dict(), file_path)

    return value_net
