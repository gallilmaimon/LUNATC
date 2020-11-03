import math
import random
import numpy as np
import tensorflow as tf

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# add base path so that can import other files from project
import os
import sys
LIB_DIR = os.path.abspath(__file__).split('src')[0]
sys.path.insert(1, LIB_DIR)

from src.Environments.SynonymEnvironment import SynonymEnvironment
from src.Environments.utils.action_utils import possible_actions
from src.Agents.utils.ReplayMemory import ReplayMemory, Transition

# configuration
from src.Config.Config import Config

# region constants
cfg = Config(LIB_DIR + "src/Config/DQN_constants.yml")
base_path = cfg.params["base_path"]
# endregion constants


class DQNNet(nn.Module):
    def __init__(self, s_shape, n_actions):
        super(DQNNet, self).__init__()
        self.linear1 = nn.Linear(s_shape, 500)
        self.relu1 = nn.LeakyReLU()

        self.linear2 = nn.Linear(500, 200)
        self.relu2 = nn.LeakyReLU()

        self.out = nn.Linear(200, n_actions)

    def forward(self, x):
        x = self.relu1(self.linear1(x))
        x = self.relu2(self.linear2(x))
        return self.out(x)


class DQNAgent:
    def __init__(self, sent_list, text_model, norm, device='cuda', mem_size=10000):
        state_shape = cfg.params["STATE_SHAPE"]
        n_actions = cfg.params["MAX_SENT_LEN"]
        self.norm = norm
        self.device = device

        sess = tf.Session()
        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

        self.env = SynonymEnvironment(cfg.params["MAX_SENT_LEN"], sent_list, sess, init_sentence=None,
                                      text_model=text_model, max_turns=cfg.params["MAX_TURNS"])

        self.policy_net = DQNNet(state_shape, n_actions).to(device)
        self.target_net = DQNNet(state_shape, n_actions).to(device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.memory = ReplayMemory(mem_size)

        # DQN parameters
        self.gamma = cfg.params['GAMMA']
        self.eps_start = cfg.params['EPS_START']
        self.eps_end = cfg.params['EPS_END']
        self.eps_decay = cfg.params['EPS_DECAY']
        self.batch_size = cfg.params['BATCH_SIZE']
        self.target_update = cfg.params['TARGET_UPDATE']

        self.steps_done = 0
        self.rewards = []

    def select_action(self, s, legal_actions):
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if random.random() > eps_threshold:
            with torch.no_grad():
                # return the action with the largest expected reward from within the legal actions
                return torch.tensor(legal_actions[self.policy_net(s)[:, legal_actions].max(1)[1]], device=self.device,
                                    dtype=torch.long).view(1, 1)
                # return self.policy_net(s).max(1)[1].view(1, 1)
        else:
            # return torch.tensor([[random.randrange(cfg.params["MAX_SENT_LEN"])]], device=device, dtype=torch.long)
            return torch.tensor([random.sample(legal_actions, 1)], device=self.device, dtype=torch.long)

    def _optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for detailed explanation). This converts
        # batch-array of Transitions to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        legal_batch = torch.cat(batch.legal_moves)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken. These are the
        # actions which would've been taken for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states. Expected values of actions for non_final_next_states are computed
        # based on the "older" target_net; selecting their best reward with max(1)[0]. This is merged based on the mask,
        # such that we'll have either the expected state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        # next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        # For double DQNNet - variant
        chosen_actions = (self.policy_net(non_final_next_states)+(-1*np.inf*(legal_batch[non_final_mask]))).argmax(1).detach().view(-1, 1)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, chosen_actions).view(-1)

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def train_model(self, num_episodes):
        for i_episode in range(num_episodes):
            # Initialize the environment and state
            s = self.env.reset()
            done = False
            s = torch.Tensor(s).to(self.device).view(1, -1)
            s = self.norm.normalize(s)
            tot_reward = 0
            while not done:
                # Select and perform an action
                action = self.select_action(s, self.env.legal_moves)
                s_new, reward, done, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.device, dtype=torch.float32)
                s_new = self.norm.normalize(torch.Tensor(s_new).to(self.device).view(1, -1)) if not done else None
                # s_new = torch.Tensor(s_new).to(device).view(1, -1) if not done else None
                tot_reward += reward

                legal_moves = torch.zeros([1, cfg.params["MAX_SENT_LEN"]], dtype=bool)
                legal_moves[:, possible_actions(self.env.state)] = True

                # Store the transition in memory
                self.memory.push(s, action, s_new, reward, legal_moves.to(self.device))

                # Move to the next state
                s = s_new

                # Perform one step of the optimization (on the target network)
                self._optimize_model()
                if done:
                    self.rewards.append(tot_reward)
                    self.env.render()
                    print("Ep:", i_episode, "| Ep_r: %.5f" % tot_reward)
                    break

            # Update the target network, copying all weights and biases in DQNNet
            if i_episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

