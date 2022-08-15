import math
import random
import pickle
import tensorflow as tf

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_value_

# add base path so that can import other files from project
import os
import sys
LIB_DIR = os.path.abspath(__file__).split('src')[0]
sys.path.insert(1, LIB_DIR)

from src.Environments.SynonymEnvironment import SynonymEnvironment
from src.Environments.utils.action_utils import possible_actions
from src.Attacks.Agents.Memory.ReplayMemory import ReplayMemory, Transition
from src.Attacks.Agents.Memory.PrioritisedMemory import PrioritisedMemory

# configuration
from src.Config.Config import Config

# region constants
cfg = Config(LIB_DIR + "src/Config/constants.yml")
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
    def __init__(self, sent_list, text_model, n_actions, norm=None, device='cuda', mem_size=10000, test_mode=False):
        state_shape = cfg.params["STATE_SHAPE"]
        self.n_actions = n_actions
        self.norm = norm
        self.device = device
        self.test_mode = test_mode

        sess = tf.Session()
        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

        self.env = SynonymEnvironment(n_actions, sent_list, sess, init_sentence=None, text_model=text_model,
                                      max_turns=cfg.params["MAX_TURNS"])

        self.policy_net = DQNNet(state_shape, n_actions).to(device)
        self.target_net = DQNNet(state_shape, n_actions).to(device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.params["LEARNING_RATE"])
        self.memory = self.memory = PrioritisedMemory(mem_size) if cfg.params["MEM_TYPE"] == 'priority' else ReplayMemory(mem_size)

        # DQN parameters
        self.gamma = cfg.params['GAMMA']
        self.eps_start = cfg.params['EPS_START']
        self.eps_end = cfg.params['EPS_END']
        self.eps_decay = cfg.params['EPS_DECAY']
        self.batch_size = cfg.params['BATCH_SIZE']
        self.target_update = cfg.params['TARGET_UPDATE']

        self.steps_done = 0
        self.rewards = []
        self.init_states = []
        self.final_states = []

    def select_action(self, s, legal_actions):
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if random.random() > eps_threshold or self.test_mode:
            with torch.no_grad():
                # return the action with the largest expected reward from within the legal actions
                return torch.tensor(legal_actions[self.policy_net(s)[:, legal_actions].max(1)[1]], device=self.device,
                                    dtype=torch.long).view(1, 1)
        else:
            return torch.tensor([random.sample(legal_actions, 1)], device=self.device, dtype=torch.long)

    def _optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = None
        if type(self.memory) == ReplayMemory:
            transitions = self.memory.sample(self.batch_size)
        elif type(self.memory) == PrioritisedMemory:
            transitions, idx, is_weight = self.memory.sample(self.batch_size)

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
        chosen_actions = (self.target_net(non_final_next_states) - (1e5 * (~legal_batch[non_final_mask]))).argmax(1).detach().view(-1, 1)  # this chooses only legal actions
        next_state_values[non_final_mask] = self.policy_net(non_final_next_states).gather(1, chosen_actions).view(-1)

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        if type(self.memory) == ReplayMemory:
            weighted_loss = loss.mean()
        elif type(self.memory) == PrioritisedMemory:
            weighted_loss = (torch.FloatTensor(is_weight).to(self.device) * loss.view(-1)).mean()

            # update priorities
            loss2 = torch.clone(loss).view(-1).detach().cpu().numpy()
            for j in range(len(loss2)):
                self.memory.update(idx[j], loss2[j])

        # Optimize the model
        self.optimizer.zero_grad()
        weighted_loss.backward()
        clip_grad_value_(self.policy_net.parameters(), 1)

        self.optimizer.step()

    def save_agent(self, path, slim=False):
        os.makedirs(path, exist_ok=True)
        # save models and optimiser
        torch.save(self.policy_net.state_dict(), path + '/policy.pth')
        torch.save(self.target_net.state_dict(), path + '/target.pth')
        torch.save(self.optimizer.state_dict(), path + '/optim.pth')

        # save parameters
        param_dict = {'gamma': self.gamma, 'eps_start': self.eps_start, 'eps_end': self.eps_end,
                      'eps_decay': self.eps_decay, 'batch_size': self.batch_size, 'target_update': self.target_update,
                      'steps_done': self.steps_done, 'n_actions': self.n_actions, 'device': self.device}
        with open(path + '/parameters.pkl', 'wb') as f:
            pickle.dump(param_dict, f)

        # save normaliser
        with open(path + '/norm.pkl', 'wb') as f:
            pickle.dump(self.norm, f)

        # save memory
        if slim:
            memory = PrioritisedMemory(self.memory.capacity) if cfg.params["MEM_TYPE"] == 'priority' \
                else ReplayMemory(self.memory.capacity)
        else:
            memory = self.memory
        with open(path + '/memory.pkl', 'wb') as f:
            pickle.dump(memory, f)

    def load_agent(self, path):
        # load models and optimiser
        self.policy_net.load_state_dict(torch.load(path + '/policy.pth'))
        self.target_net.load_state_dict(torch.load(path + '/target.pth'))
        self.optimizer.load_state_dict(torch.load(path + '/optim.pth'))

        # load parameters
        with open(path + '/parameters.pkl', 'rb') as f:
            params = pickle.load(f)
        self.gamma, self.eps_start, self.eps_end = params['gamma'], params['eps_start'], params['eps_end']
        self.eps_decay, self.batch_size, self.target_update = params['eps_decay'], params['batch_size'], params['target_update']
        self.steps_done, self.n_actions = params['steps_done'], params['n_actions']
        self.device = params['device']

        # load normaliser
        with open(path + '/norm.pkl', 'rb') as f:
            self.norm = pickle.load(f)

        # load memory
        with open(path + '/memory.pkl', 'rb') as f:
            self.memory = pickle.load(f)

    def _get_td_error(self, s, a, s_new, r, legal_moves):
        with torch.no_grad():
            pred_q = self.policy_net(s).gather(1, a)

            if s_new is None:
                next_q = 0
            else:
                chosen_a = (self.target_net(s_new) - 1e5 * (~legal_moves)).argmax().detach().view(-1, 1)  # this chooses from legal actions only
                next_q = self.policy_net(s_new).gather(1, chosen_a)
            calc_q = r + self.gamma * next_q
            return F.smooth_l1_loss(calc_q.view(-1), pred_q.view(-1)).cpu().numpy()

    def train_model(self, num_episodes, optimise=True):
        for i_episode in range(num_episodes):
            # Initialize the environment and state
            s = self.env.reset()

            self.init_states.append(self.env.state)
            tot_reward = torch.tensor([.0]).to(self.device)

            if len(self.env.legal_moves) == 0:  # if there are no legal actions the round is done
                done = True
            else:
                done = False

                # get embedded action representation and embed state
                s = torch.Tensor(s).to(self.device).view(1, -1)
                s = self.norm.normalize(s) if self.norm is not None else s

            while not done:
                # Select and perform an action
                action = self.select_action(s, self.env.legal_moves)
                s_new, reward, done, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.device, dtype=torch.float32)
                if self.norm is not None:
                    s_new = self.norm.normalize(torch.Tensor(s_new).to(self.device).view(1, -1)) if not done else None
                else:
                    s_new = torch.Tensor(s_new).to(self.device).view(1, -1) if not done else None
                tot_reward += reward

                legal_moves = torch.zeros([1, self.n_actions], dtype=bool)
                legal_moves[:, possible_actions(self.env.state)] = True

                # Store the transition in memory
                if type(self.memory) == ReplayMemory:
                    self.memory.push(s, action, s_new, reward, legal_moves.to(self.device), None)
                elif type(self.memory) == PrioritisedMemory:
                    td_err = self._get_td_error(s, action, s_new, reward, legal_moves.to(self.device))
                    self.memory.add(td_err, (s, action, s_new, reward, legal_moves.to(self.device), None))

                # Move to the next state
                s = s_new

                # Perform one step of the optimization (on the target network)
                self._optimize_model() if optimise else ''

            self.rewards.append(tot_reward.item())
            self.final_states.append(self.env.state)
            self.env.render()
            print("Ep:", i_episode, "| Ep_r: %.5f" % tot_reward)

            # Update the target network, copying all weights and biases in DQNNet
            if i_episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
