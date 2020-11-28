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

from src.Environments.ContinuousSynonymEnvironment import ContinuousSynonymEnvironment
from src.Agents.Memory.ReplayMemory import ReplayMemory, Transition
from src.Agents.Memory.PrioritisedMemory import PrioritisedMemory
from src.TextModels.text_model_utils import load_embedding_dict

# configuration
from src.Config.Config import Config

# region constants
cfg = Config(LIB_DIR + "src/Config/DQN_constants.yml")
base_path = cfg.params["base_path"]
# endregion constants


class ContinuousDQNNet(nn.Module):
    def __init__(self, s_shape, a_shape):
        super(ContinuousDQNNet, self).__init__()
        self.linear1 = nn.Linear(s_shape + a_shape, 500)
        self.relu1 = nn.LeakyReLU()

        self.linear2 = nn.Linear(500, 200)
        self.relu2 = nn.LeakyReLU()

        # self.linear3 = nn.Linear(200, 100)
        # self.relu3 = nn.LeakyReLU()
        #
        # self.linear4 = nn.Linear(100, 32)
        # self.relu4 = nn.LeakyReLU()

        self.out = nn.Linear(200, 1)

    def forward(self, x):
        x = self.relu1(self.linear1(x))
        x = self.relu2(self.linear2(x))
        # x = self.relu3(self.linear3(x))
        # x = self.relu4(self.linear4(x))
        return self.out(x)


class ContinuousDQNAgent:
    def __init__(self, sent_list, text_model, n_actions, norm=None, device='cuda', mem_size=10000):
        self.state_shape = cfg.params["STATE_SHAPE"]
        self.n_actions = n_actions
        self.action_shape = 200  # TODO: make not constant
        self.norm = norm
        self.device = device

        sess = tf.Session()
        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

        self.env = ContinuousSynonymEnvironment(n_actions, sent_list, sess, init_sentence=None, text_model=text_model,
                                                max_turns=cfg.params["MAX_TURNS"])

        self.policy_net = ContinuousDQNNet(self.state_shape, self.action_shape).to(device)
        self.target_net = ContinuousDQNNet(self.state_shape, self.action_shape).to(device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.params["LEARNING_RATE"])

        self.memory = PrioritisedMemory(mem_size) if cfg.params["MEM_TYPE"] == 'priority' else ReplayMemory(mem_size)

        # Glove embeddings for action embedding
        glove_path = '/resources/word_vectors/glove.6B.200d.txt'  # TODO: make configurable
        self.word2vec = load_embedding_dict(LIB_DIR + glove_path, torch.rand((1, 200)))  # TODO: make configurable

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

    def select_action(self, s, legal_actions, action_embeddings):
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if random.random() > eps_threshold:
            with torch.no_grad():
                s = s.repeat(len(legal_actions), 1)
                s_a = torch.cat([s, action_embeddings], axis=1)
                # return the action with the largest expected reward from within the legal actions
                best_a = self.policy_net(s_a).argmax(0)
                return torch.tensor(legal_actions[best_a], device=self.device, dtype=torch.long).view(1, 1), \
                       action_embeddings[best_a[0], :]

        else:
            ind = random.randint(0, len(legal_actions) - 1)
            return torch.tensor([[legal_actions[ind]]], device=self.device, dtype=torch.long), action_embeddings[ind, :]

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

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device,
                                      dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        next_state_actions = torch.cat([s for s in batch.emb_next_action if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        legal_batch = torch.cat(batch.legal_moves)[non_final_mask]

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken. These are the
        # actions which would've been taken for each batch state according to policy_net
        state_action_values = self.policy_net(torch.cat([state_batch, action_batch], axis=1))

        # Compute V(s_{t+1}) for all next states. Expected values of actions for non_final_next_states are computed
        # based on the "older" target_net; selecting their best reward with max(1)[0]. This is merged based on the mask,
        # such that we'll have either the expected state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        legal_count = legal_batch.sum(axis=1)
        stacked_next_state = non_final_next_states.repeat((1, self.n_actions)).view(-1, self.state_shape)

        stacked_input = torch.cat([stacked_next_state, next_state_actions], dim=1)
        net_out = self.policy_net(stacked_input).view(-1, self.n_actions)

        # TODO: make more efficient
        mask = torch.zeros(net_out.shape[0], net_out.shape[1] + 1, dtype=net_out.dtype, device=net_out.device)
        mask[(torch.arange(net_out.shape[0]), legal_count)] = 1
        mask = mask.cumsum(dim=1)[:, :-1]
        mask = mask*torch.tensor(-1*float("inf"))
        mask[mask != mask] = 0
        net_out += mask

        chosen_actions = net_out.argmax(1) + torch.arange(start=0, end=self.n_actions*len(non_final_next_states)-1,
                                                          step=self.n_actions, device=self.device)

        # input selected actions to target net
        target_net_input = torch.cat([non_final_next_states, next_state_actions.index_select(0, chosen_actions)], dim=1)
        next_state_values[non_final_mask] = self.target_net(target_net_input).view(-1)

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1), reduction='none')

        if type(self.memory) == ReplayMemory:
            weighted_loss = loss.mean()
        elif type(self.memory) == PrioritisedMemory:
            weighted_loss = (torch.FloatTensor(is_weight).to(self.device) * loss.view(-1)).mean()
            # update priorities
            loss2 = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1), reduction='none').view(-1).detach().cpu().numpy()
            for j in range(len(loss2)):
                self.memory.update(idx[j], loss2[j])

        # Optimize the model
        self.optimizer.zero_grad()
        weighted_loss.backward()
        # for param in self.policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def _get_embedded_actions(self, text, legal_moves):
        if type(legal_moves) == list:
            if len(legal_moves) == 0:
                return None
            words = np.array(text.split())[legal_moves]
        else:
            words = np.array(text.split())[legal_moves.cpu()[0]]
        return torch.cat([self.word2vec[word] for word in words]).to(self.device)

    # def _get_embedded_actions(self, text, legal_moves):
    #     if len(legal_moves) == 0:
    #         return None
    #     mask = torch.zeros(len(legal_moves), 100, device=self.device)
    #     mask[(torch.arange(len(legal_moves)), legal_moves)] = 1
    #     return mask

    def _get_td_error(self, s, embedded_a, s_new, r, new_legal_embedded_a):
        with torch.no_grad():
            pred_q = self.policy_net(torch.cat([s, embedded_a], axis=1))

            if s_new is None:
                calc_q = r
            else:
                net_inp = torch.cat([s_new.repeat((len(new_legal_embedded_a), 1)), new_legal_embedded_a], axis=1)
                chosen_a = self.target_net(net_inp).view(-1).argmax()
                next_q = self.policy_net(torch.cat([s_new, new_legal_embedded_a[chosen_a].unsqueeze(0)], axis=1))
                calc_q = r + self.gamma * next_q

            return F.smooth_l1_loss(calc_q, pred_q).cpu().numpy()

    def train_model(self, num_episodes):
        for i_episode in range(num_episodes):
            # Initialize the environment and state
            s = self.env.reset()
            self.init_states.append(self.env.state)
            done = False
            tot_reward = 0

            # get embedded action representation and embed state
            embedded_a = self._get_embedded_actions(s, self.env.legal_moves)
            s = self.env.get_embedded_state(s).to(self.device).view(1, -1)
            s = self.norm.normalize(s) if self.norm is not None else s

            while not done:
                # Select and perform an action
                action, emb_a = self.select_action(s, self.env.legal_moves, embedded_a)

                s_new, reward, done, _ = self.env.step(action.item())

                new_emb_a = self._get_embedded_actions(s_new, self.env.legal_moves) if not done else None
                new_emb_a_pad = torch.cat([new_emb_a, torch.zeros((self.n_actions - len(new_emb_a), self.action_shape),
                                                                  device=self.device)]) if not done else None
                reward = torch.tensor([reward], device=self.device, dtype=torch.float32)
                if self.norm is not None:
                    s_new = self.norm.normalize(self.env.get_embedded_state(s_new).to(self.device).view(1, -1)) if not done else None
                else:
                    s_new = self.env.get_embedded_state(s_new).view(1, -1) if not done else None
                tot_reward += reward

                legal_moves = torch.zeros([1, self.n_actions], dtype=bool)
                legal_moves[:, self.env.legal_moves] = True
                legal_moves = legal_moves.to(self.device)

                # Store the transition in memory
                if type(self.memory) == ReplayMemory:
                    self.memory.push(s, emb_a.unsqueeze(0), s_new, reward, legal_moves, new_emb_a_pad)
                elif type(self.memory) == PrioritisedMemory:
                    td_err = self._get_td_error(s, emb_a.unsqueeze(0), s_new, reward, new_emb_a)
                    self.memory.add(td_err, (s, emb_a.unsqueeze(0), s_new, reward, legal_moves, new_emb_a_pad))

                # Move to the next state
                s = s_new
                embedded_a = new_emb_a

                # Perform one step of the optimization
                self._optimize_model()
                if done:
                    self.final_states.append(self.env.state)
                    self.rewards.append(tot_reward.item())
                    self.env.render()
                    print("Ep:", i_episode, "| Ep_r: %.5f" % tot_reward)
                    break

            # Update the target network, copying all weights and biases in DQNNet
            if i_episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
