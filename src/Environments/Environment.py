from abc import ABC, abstractmethod
import numpy as np
from src.Environments.utils.action_utils import get_similarity
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


class Environment(ABC):
    @abstractmethod
    def __init__(self, max_sent_len, sent_list, sess, init_sentence=None, text_model=None, max_turns=30,
                 ppl_diff=False, device='cuda', embed_states=True):
        self.text_model = text_model
        self.ppl_diff = ppl_diff
        self.lm = GPT2LMHeadModel.from_pretrained('gpt2').to(device).half() if ppl_diff else None
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2') if ppl_diff else None
        if ppl_diff:
            self.tokenizer.padding_side = "right"
            self.tokenizer.pad_token = self.tokenizer.eos_token  # to avoid an error

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
        self.embed_states = embed_states
        self.reset(init_sentence)

    @abstractmethod
    def reset(self, init_sentence=None):
        # end previous episode
        if self.score is not None:
            self.history.append(self.score)

        # reset score
        self.score = 0

        if init_sentence is None:
            # np.random.seed()
            self.init_sentence = np.random.choice(self.sent_list)
        else:
            self.init_sentence = init_sentence

        self.state = self.init_sentence
        self.cur_prob = self.text_model.predict_proba(self.init_sentence)[0]
        self.original_class = np.argmax(self.cur_prob)
        self.turn = 0
        if self.embed_states:
            return self.get_embedded_state(self.state)
        return self.state

    @abstractmethod
    def render(self):
        print('initial sentence', self.init_sentence, flush=True)
        print('state', self.state, flush=True)
        print('moves made: ', self.turn, flush=True)

    @abstractmethod
    def r(self, a):
        if self.ppl_diff:
            new_s, ppl_diff = self.delta(self.state, a)
        else:
            new_s = self.delta(self.state, a)
            ppl_diff = 0

        ppl_diff = min(ppl_diff, 0)

        # actions with no effect get negative reward
        if new_s == self.state:
            return -0.0001, False, new_s
        new_proba = self.text_model.predict_proba(new_s)[0]
        old_proba = self.cur_prob

        # the change in difference between class model outputs - the smaller the difference the closer the model to
        # changing predictions
        logit_diff = (old_proba[self.original_class] - old_proba[1 - self.original_class]) - \
                     max((new_proba[self.original_class] - new_proba[1 - self.original_class]), 0)

        self.cur_prob = new_proba
        if np.argmax(new_proba) == self.original_class:
            return logit_diff + 0.2 * ppl_diff, False, new_s

        return 100 * get_similarity([self.init_sentence, new_s], self.sess)[0] + logit_diff + 0.2 * ppl_diff, True, new_s

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
    def get_embedded_state(self, state, ret_type='numpy'):
        if ret_type == 'pt':
            return self.text_model.embed(state)
        return self.text_model.embed(state).cpu().numpy()[0]
