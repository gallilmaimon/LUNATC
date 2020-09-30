from abc import ABC, abstractmethod
import numpy as np
from text_xai.Environments.utils.action_utils import get_similarity


class Environment(ABC):
    @abstractmethod
    def __init__(self, max_sent_len, sent_list, sess, init_sentence=None, lang_model=None, max_turns=30):
        self.lang_model = lang_model
        self.max_sent_len = max_sent_len
        self.history = []
        self.score = None
        self.state = None
        self.original_class = None
        self.init_sentence = None
        self.sent_list = sent_list
        self.legal_moves = None
        self.sess = sess
        self.max_turns = max_turns
        self.turn = 0
        self.cur_prob = None
        self.reset(init_sentence)

    @abstractmethod
    def reset(self, init_sentence=None):
        # end previous episode
        if self.score is not None:
            self.history.append(self.score)

        # reset score
        self.score = 0

        if init_sentence is None:
            np.random.seed()
            self.init_sentence = np.random.choice(self.sent_list)
        else:
            self.init_sentence = init_sentence

        self.state = self.init_sentence
        self.cur_prob = self.lang_model.predict_proba(self.init_sentence)[0]
        self.original_class = np.argmax(self.cur_prob)
        self.turn = 0
        return self.get_embedded_state(self.state)

    @abstractmethod
    def render(self):
        print('initial sentence', self.init_sentence)
        print('state', self.state)
        print('moves made: ', self.turn)

    @abstractmethod
    def r(self, a):
        new_s = self.delta(self.state, a)

        # actions with no effect get negative reward
        if new_s == self.state:
            return -0.0001, False, new_s
        new_proba = self.lang_model.predict_proba(new_s)[0]
        old_proba = self.cur_prob

        # the change in difference between class model outputs - the smaller the difference the closer the model to
        # changing predictions
        logit_diff = (old_proba[self.original_class] - old_proba[1 - self.original_class]) - \
                     (new_proba[self.original_class] - new_proba[1 - self.original_class])

        self.cur_prob = new_proba
        if np.argmax(new_proba) == self.original_class:
            return logit_diff, False, new_s

        return 100 * get_similarity([self.init_sentence, new_s], self.sess)[0] + logit_diff, True, new_s

    @abstractmethod
    def get_legal_moves(self):
        return self.legal_moves

    @staticmethod
    @abstractmethod
    def delta(s, a):
        return None

    @abstractmethod
    def step(self, a):
        return None

    @abstractmethod
    def get_embedded_state(self, state):
        return self.lang_model.embed(state).cpu().numpy()[0]
