import time
import numpy as np
import os
from torch import multiprocessing as mp

from text_xai.Agents.utils.utils import record
from text_xai.Environments.SynonymEnvironment import SynonymEnvironment
from text_xai.Environments.SynonymDeleteEnvironment import SynonymDeleteEnvironment
from text_xai.Environments.SynonymMisspellEnvironement import SynonymMisspellEnvironment
from text_xai.Config.Config import Config


# region constants
LIB_DIR = os.path.abspath(__file__).split('text_xai')[0]
cfg = Config(LIB_DIR + "text_xai/Config/constants.yml")
base_path = cfg.params["base_path"]
# endregion constants


class SearchAgentWorker(mp.Process):
    def __init__(self, sent_list, global_ep, name, n, text_model, max_sent_len, num_episodes):
        super(SearchAgentWorker, self).__init__()
        self.name = 'w%i' % name
        self.g_ep = global_ep
        self.n = n
        self.num_episodes = num_episodes
        self.early_stopping = cfg.params["EARLY_STOPPING"]
        self.calc_oracle_usage = cfg.params["CALCULATE_ORACLE_USE"]

        if cfg.params["ENV_TYPE"] == 'Synonym':
            self.env = SynonymEnvironment(max_sent_len, sent_list, None, init_sentence=None,
                                          text_model=text_model, max_turns=cfg.params["MAX_TURNS"])
        elif cfg.params["ENV_TYPE"] == 'SynonymDelete':
            self.env = SynonymDeleteEnvironment(max_sent_len, sent_list, None, init_sentence=None,
                                                text_model=text_model, max_turns=cfg.params["MAX_TURNS"])
        elif cfg.params["ENV_TYPE"] == 'SynonymMisspell':
            self.env = SynonymMisspellEnvironment(max_sent_len, sent_list, None, init_sentence=None,
                                                  text_model=text_model, max_turns=cfg.params["MAX_TURNS"])
        else:
            print("Unsupported ENV_TYPE !")
            exit(1)

    def run(self):
        import tensorflow as tf
        self.env.sess = tf.Session()
        self.env.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

        total_step = 1
        while self.g_ep.value < self.num_episodes:
            _ = self.env.reset()
            ep_r = 0.
            t1 = time.time()
            while True:
                np.random.seed()
                a = np.random.choice(self.env.get_legal_moves())

                if self.calc_oracle_usage:
                    with open(f'{base_path}_search_results/{self.n}/{self.name}_states.txt', 'a+') as f:
                        f.write(f'{self.env.state}\n')

                _, r, done, _ = self.env.step(a)
                ep_r += r
                if done:  # done and print information
                    if self.name in ['w0', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7']:
                        self.env.render()
                    total_step = 1
                    record(self.g_ep, ep_r, self.name)
                    break

                total_step += 1
            print(f'episode time is: {time.time() - t1}')
            with open(f'{base_path}_search_results/{self.n}/{self.name}.txt', 'a+') as f:
                f.write(f'{self.env.init_sentence}##g##{self.g_ep.value}##g##{ep_r}##g##{self.env.state}\n')

            # Early stopping
            if self.early_stopping is not None and ep_r > self.early_stopping:
                with self.g_ep.get_lock():
                    self.g_ep.value = self.num_episodes
                break
