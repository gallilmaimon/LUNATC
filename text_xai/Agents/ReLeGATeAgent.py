import time

import numpy as np
import torch
from torch import nn as nn, multiprocessing as mp
from torch.nn import functional as F

from text_xai.Agents.utils.utils import set_init, v_wrap, push_and_pull, synced_update, record
from text_xai.Environments.SynonymEnvironment import SynonymEnvironment
from text_xai.Environments.SynonymDeleteEnvironment import SynonymDeleteEnvironment
from text_xai.Environments.SynonymMisspellEnvironement import SynonymMisspellEnvironment

from text_xai.Config.Config import Config
import os

# region constants
LIB_DIR = os.path.abspath(__file__).split('text_xai')[0]
cfg = Config(LIB_DIR + "text_xai/Config/constants.yml")
base_path = cfg.params["base_path"]
# endregion constants


class Normalizer:
    def __init__(self, num_inputs, norm_rounds):
        print("init normaliser")
        self.n = torch.zeros(1, num_inputs)
        self.mean = torch.zeros(1, num_inputs)
        self.mean_diff = torch.zeros(1, num_inputs)
        self.var = torch.ones(1, num_inputs)
        self.norm_rounds = norm_rounds

    def observe(self, x):
        self.n += 1.
        if self.n[0][0] < self.norm_rounds:
            last_mean = self.mean.clone()
            self.mean += (x-self.mean)/self.n
            self.mean_diff += (x-last_mean)*(x-self.mean)
            self.var = torch.clamp(self.mean_diff/self.n, min=1e-8)

    def normalize(self, inputs):
        obs_std = torch.sqrt(self.var)
        return (inputs - self.mean)/obs_std


class ReLeGATeAgentNet(nn.Module):
    def __init__(self, s_dim, a_dim, norm, lock):
        super(ReLeGATeAgentNet, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.norm = norm
        self.lock = lock
        self.l1 = nn.Linear(s_dim, 500)
        self.l2 = nn.Linear(500, 200)
        self.logit_out = nn.Linear(200, a_dim)
        self.v_out = nn.Linear(200, 1)
        set_init([self.l1, self.l2, self.logit_out, self.v_out])
        self.distribution = torch.distributions.Categorical
        self.entropy_alpha = cfg.params["ENTROPY_ALPHA"]

    def forward(self, x):
        # normalise data
        if self.norm is not None:
            if x.shape[0] == 1:
                self.lock.acquire()  # TODO: make locks only apply to when parameters are changing and not for normalising once fixed
                self.norm.observe(x) if self.training else ''
                x = self.norm.normalize(x)
                self.lock.release()
            else:
                for j in range(x.shape[0]):  # TODO: MAKE MORE EFFICIENT
                    self.lock.acquire()
                    self.norm.observe(x[j]) if self.training else ''
                    x[j] = self.norm.normalize(x[j])
                    self.lock.release()
        l1 = torch.tanh(self.l1(x))
        l2 = torch.tanh(self.l2(l1))
        logits = self.logit_out(l2)
        values = self.v_out(l2)
        return logits, values

    def choose_action(self, s, legal_moves):
        self.eval()
        # without this line all processes choose the same action for the same distribution
        torch.manual_seed(np.random.randint(0, 10000000000))
        with torch.no_grad():
            logits, _ = self.forward(s)
        prob = F.softmax(logits[0][legal_moves], dim=0).data  # sample only legal moves
        m = self.distribution(prob)
        act = m.sample().numpy()
        return np.array(legal_moves[act])

    def loss_func(self, s, a, r_t):
        self.train()
        logits, values = self.forward(s)
        advantage = r_t - values
        critic_loss = advantage.pow(2)

        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * advantage.detach().squeeze()
        actor_loss = -exp_v
        total_loss = (critic_loss + actor_loss).mean()

        # add entropy loss for better exploration
        entropy_loss = m.entropy().mean()

        return total_loss - 1 * self.entropy_alpha * entropy_loss


class ReLeGATeAgentWorker(mp.Process):
    def __init__(self, gnet: ReLeGATeAgentNet, opt, sent_list, global_ep, name, n, lang_model,
                 max_sent_len, num_episodes, barrier, epoch=None, train=True, sync_barrier=None, sync_event=None,
                 v_target_queue=None, a_queue=None, s_queue=None):
        super(ReLeGATeAgentWorker, self).__init__()
        self.name = 'w%i' % name
        if epoch is not None:
            self.name = f'{epoch}/w{name}'
        self.epoch = epoch
        self.g_ep = global_ep
        self.gnet, self.opt = gnet, opt
        self.lnet = ReLeGATeAgentNet(gnet.s_dim, gnet.a_dim, gnet.norm, gnet.lock)    # local network
        self.train = train

        self.calc_oracle_usage = cfg.params["CALCULATE_ORACLE_USE"]

        # load previous worker net & opt weights if we are training and this is not the first epoch (and they exist)
        if self.train and epoch != 0 and epoch is not None and os.path.exists(f'{base_path}_agent_{self.name[-1]}.pth'):
            self.lnet.load_state_dict(torch.load(f'{base_path}_agent_{self.name[-1]}.pth'))
            self.opt.load_state_dict(torch.load(f'{base_path}_optimiser_{self.name[-1]}.pth'))
        else:
            self.lnet.load_state_dict(self.gnet.state_dict())  # initialise local network by global
        self.n = n

        self.num_episodes = num_episodes

        # a3c training parameters
        self.gamma = cfg.params["GAMMA"]
        self.global_update_iter = cfg.params["UPDATE_GLOBAL_ITER"]
        self.early_stopping = cfg.params["EARLY_STOPPING"]
        self.sync_updates = cfg.params["SYNC_UPDATE"]

        # initialise selected environment type
        if cfg.params["ENV_TYPE"] == 'Synonym':
            self.env = SynonymEnvironment(max_sent_len, sent_list, None, init_sentence=None,
                                          lang_model=lang_model, max_turns=cfg.params["MAX_TURNS"])
        elif cfg.params["ENV_TYPE"] == 'SynonymDelete':
            self.env = SynonymDeleteEnvironment(max_sent_len, sent_list, None, init_sentence=None,
                                                lang_model=lang_model, max_turns=cfg.params["MAX_TURNS"])
        elif cfg.params["ENV_TYPE"] == 'SynonymMisspell':
            self.env = SynonymMisspellEnvironment(max_sent_len, sent_list, None, init_sentence=None,
                                                  lang_model=lang_model, max_turns=cfg.params["MAX_TURNS"])
        else:
            print("Unsupported ENV_TYPE !")
            exit(1)

        self.barrier = barrier

        # for A2C variant
        self.sync_barrier = sync_barrier
        self.sync_event = sync_event
        self.a_queue = a_queue
        self.s_queue = s_queue
        self.v_target_queue = v_target_queue

    def run(self):
        # synchronise the start of the workers - slower, but can be better exploration
        if self.barrier is not None:
            print(f"{self.name} is waiting")
            t1 = time.time()
            self.barrier.wait()
            print(f"{self.name} finished waiting, {time.time()-t1}")
        import tensorflow as tf
        self.env.sess = tf.Session()
        self.env.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

        total_step = 1
        while self.g_ep.value < self.num_episodes:
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            t1 = time.time()
            while True:
                a = self.lnet.choose_action(v_wrap(s[None, :]), self.env.get_legal_moves())
                if self.calc_oracle_usage:
                    with open(f'{base_path}_relax_results/{self.n}/{self.name}_states.txt', 'a+') as f:
                        f.write(f'{self.env.state}\n')

                s_, r, done, _ = self.env.step(a)
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)
                if total_step % self.global_update_iter == 0 or done:
                    if self.train:
                        # update global network and assign to local net
                        if self.sync_updates:
                            synced_update(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r,
                                          self.gamma, self.sync_barrier, self.sync_event, self.s_queue,
                                          self.v_target_queue, self.a_queue)
                        else:
                            push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r,
                                          self.gamma)

                    buffer_s, buffer_a, buffer_r = [], [], []
                    if done:  # done and print information
                        if int(self.name[-1]) in range(8):
                            self.env.render()
                        total_step = 1
                        record(self.g_ep, ep_r, self.name)
                        break
                s = s_
                total_step += 1
            print(f'episode time is: {time.time() - t1}')
            with open(f'{base_path}_relegate_results/{self.n}/{self.name}.txt', 'a+') as f:
                f.write(f'{self.env.init_sentence}##g##{self.g_ep.value}##g##{ep_r}##g##{self.env.state}\n')

            # Early stopping
            if self.early_stopping is not None and ep_r > self.early_stopping:
                with self.g_ep.get_lock():
                    self.g_ep.value = self.num_episodes
                break

        if self.train and self.epoch is not None:
            # save final agent model
            torch.save(self.lnet.state_dict(), f'{base_path}_agent_{self.name[-1]}.pth')
            torch.save(self.opt.state_dict(), f'{base_path}_optimiser_{self.name[-1]}.pth')