import math
import random
import pickle
import numpy as np
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
from src.Attacks.Agents.Memory.ReplayMemory import ReplayMemory, Transition
from src.Attacks.Agents.Memory.PrioritisedMemory import PrioritisedMemory
from src.TextModels.text_model_utils import load_embedding_dict

# configuration
from src.Config.Config import Config

# region constants
cfg = Config(LIB_DIR + "src/Config/constants.yml")
base_path = cfg.params["base_path"]
# endregion constants


class ContinuousDQNNetLarge(nn.Module):
    def __init__(self, s_shape, a_shape):
        super(ContinuousDQNNetLarge, self).__init__()
        self.linear1 = nn.Linear(s_shape + a_shape, 500)
        self.relu1 = nn.LeakyReLU()

        self.linear2 = nn.Linear(500, 200)
        self.relu2 = nn.LeakyReLU()

        self.linear3 = nn.Linear(200, 100)
        self.relu3 = nn.LeakyReLU()

        self.linear4 = nn.Linear(100, 32)
        self.relu4 = nn.LeakyReLU()

        self.linear5 = nn.Linear(32, 32)
        self.relu5 = nn.LeakyReLU()

        self.linear6 = nn.Linear(32, 32)
        self.relu6 = nn.LeakyReLU()

        self.out = nn.Linear(32, 1)

    def forward(self, x):
        x = self.relu1(self.linear1(x))
        x = self.relu2(self.linear2(x))
        x = self.relu3(self.linear3(x))
        x = self.relu4(self.linear4(x))
        x = self.relu5(self.linear5(x))
        x = self.relu6(self.linear6(x))
        return self.out(x)


class ContinuousDQNNetSmall(nn.Module):
    def __init__(self, s_shape, a_shape):
        super(ContinuousDQNNetSmall, self).__init__()
        self.linear1 = nn.Linear(s_shape + a_shape, 500)
        self.relu1 = nn.LeakyReLU()

        self.linear2 = nn.Linear(500, 200)
        self.relu2 = nn.LeakyReLU()

        self.linear3 = nn.Linear(200, 32)
        self.relu3 = nn.LeakyReLU()

        self.out = nn.Linear(32, 1)

    def forward(self, x):
        x = self.relu1(self.linear1(x))
        x = self.relu2(self.linear2(x))
        x = self.relu3(self.linear3(x))
        return self.out(x)


class ContinuousDQNAgent:
    def __init__(self, sent_list, text_model, n_actions, norm=None, device='cuda', mem_size=10000, test_mode=False):
        self.state_shape = cfg.params["STATE_SHAPE"]
        self.n_actions = n_actions
        self.action_shape = 200  # TODO: make not constant
        self.norm = norm
        self.device = device
        self.test_mode = test_mode

        sess = tf.Session()
        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

        self.env = SynonymEnvironment(n_actions, sent_list, sess, init_sentence=None, text_model=text_model,
                                      max_turns=cfg.params["MAX_TURNS"], ppl_diff=cfg.params['USE_PPL'], device=device,
                                      embed_states=False)

        net_type = ContinuousDQNNetLarge if cfg.params['ATTACK_TYPE'] == 'universal' or \
                                            cfg.params['USE_PPL'] or self.test_mode else ContinuousDQNNetSmall
        self.policy_net = net_type(self.state_shape, self.action_shape).to(device)
        self.target_net = net_type(self.state_shape, self.action_shape).to(device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.params["LEARNING_RATE"])

        self.memory = PrioritisedMemory(mem_size) if cfg.params["MEM_TYPE"] == 'priority' else ReplayMemory(mem_size)

        # Glove embeddings for action embedding
        glove_path = '/resources/word_vectors/glove.6B.200d.txt'  # TODO: make configurable
        self.word2vec = load_embedding_dict(LIB_DIR + glove_path, torch.rand((1, 200)))  # TODO: make configurable
        angle_rates = (1/10000) ** np.linspace(0, 1, 200//2)
        positions = np.arange((cfg.params['MAX_SENT_LEN']))
        angle_rads = (positions[:, np.newaxis]) * angle_rates[np.newaxis, :]
        sines = np.sin(angle_rads)
        cosines = np.cos(angle_rads)
        pos_enc = np.empty((sines.shape[0], sines.shape[1] * 2))
        pos_enc[:, 0::2] = sines
        pos_enc[:, 1::2] = cosines
        self.position_encoding = torch.tensor(pos_enc).type(torch.float32)

        # DQN parameters
        self.gamma = cfg.params['GAMMA']
        self.eps_start = cfg.params['EPS_START']
        self.eps_end = cfg.params['EPS_END']
        self.eps_decay = cfg.params['EPS_DECAY']
        self.batch_size = cfg.params['BATCH_SIZE']
        self.target_update = cfg.params['TARGET_UPDATE']
        self.policy_update = cfg.params['POLICY_UPDATE']
        self.early_stopping = cfg.params["EARLY_STOPPING"]

        self.steps_done = 0
        self.rewards = []
        self.init_states = []
        self.final_states = []

    def select_action(self, s, legal_actions, action_embeddings):
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if random.random() > eps_threshold or self.test_mode:
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
        net_out = self.target_net(stacked_input).view(-1, self.n_actions)

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
        next_state_values[non_final_mask] = self.policy_net(target_net_input).view(-1)

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1), reduction='none')

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

    def _get_embedded_actions(self, text, legal_moves):
        text = text[1] if type(text) == tuple else text
        if type(legal_moves) == list:
            if len(legal_moves) == 0:
                return None
            words = np.array(text.split())[legal_moves]
        else:
            words = np.array(text.split())[legal_moves.cpu()[0]]
        return torch.cat([self.word2vec[word] for word in words]).to(self.device) + 1 * self.position_encoding[legal_moves].to(self.device)

    def save_agent(self, path, slim=False):
        os.makedirs(path, exist_ok=True)
        # save models and optimiser
        torch.save(self.policy_net.state_dict(), path + '/policy.pth')
        torch.save(self.target_net.state_dict(), path + '/target.pth')
        torch.save(self.optimizer.state_dict(), path + '/optim.pth')

        # save parameters
        param_dict = {'gamma': self.gamma, 'eps_start': self.eps_start, 'eps_end': self.eps_end,
                      'eps_decay': self.eps_decay, 'batch_size': self.batch_size, 'target_update': self.target_update,
                      'steps_done': self.steps_done, 'state_shape': self.state_shape, 'n_actions': self.n_actions,
                      'action_shape': self.action_shape, 'device': self.device}
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
        self.steps_done, self.state_shape, self.n_actions = params['steps_done'], params['state_shape'], params['n_actions']
        self.action_shape, self.device = params['action_shape'], params['device']

        # load normaliser
        with open(path + '/norm.pkl', 'rb') as f:
            self.norm = pickle.load(f)

        # load memory
        with open(path + '/memory.pkl', 'rb') as f:
            self.memory = pickle.load(f)

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

            return F.smooth_l1_loss(calc_q.view(-1), pred_q.view(-1)).cpu().numpy()

    def train_model(self, num_episodes, optimise=True):
        update_count = 0
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
                embedded_a = self._get_embedded_actions(s, self.env.legal_moves)
                s = self.env.get_embedded_state(s, ret_type='pt').to(self.device).view(1, -1)
                s = self.norm.normalize(s) if self.norm is not None else s

            while not done:
                update_count += 1
                # Select and perform an action
                action, emb_a = self.select_action(s, self.env.legal_moves, embedded_a)

                s_new, reward, done, _ = self.env.step(action.item())

                new_emb_a = self._get_embedded_actions(s_new, self.env.legal_moves) if not done else None
                new_emb_a_pad = torch.cat([new_emb_a, torch.zeros((self.n_actions - len(new_emb_a), self.action_shape),
                                                                  device=self.device)]) if not done else None
                reward = torch.tensor([reward], device=self.device, dtype=torch.float32)
                if self.norm is not None:
                    s_new = self.norm.normalize(self.env.get_embedded_state(s_new, ret_type='pt').to(self.device).view(1, -1)) if not done else None
                else:
                    s_new = self.env.get_embedded_state(s_new, ret_type='pt').view(1, -1) if not done else None
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
                self._optimize_model() if optimise and update_count % self.policy_update == 0 else ''

            self.final_states.append(self.env.state)
            self.rewards.append(tot_reward.item())
            self.env.render()
            print("Ep:", i_episode, "| Ep_r: %.5f" % tot_reward)
            # early stopping if the wanted reward was achieved
            if tot_reward.item() > self.early_stopping:
                return

            # Update the target network, copying all weights and biases in DQNNet
            if i_episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
