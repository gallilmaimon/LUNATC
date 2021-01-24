# imports
import numpy as np
from copy import deepcopy as copy
from src.Environments.utils.action_utils import replace_with_synonym_perplexity, replace_with_synonym, replace_with_synonym_greedy, possible_actions

from src.Environments.Environment import Environment


class ContinuousSynonymEnvironment(Environment):
    def __init__(self, max_sent_len, sent_list, sess, init_sentence=None, text_model=None, max_turns=30, ppl_diff=False,
                 device='cuda'):
        super().__init__(max_sent_len, sent_list, sess, init_sentence, text_model, max_turns, ppl_diff, device)

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

        self.legal_moves = possible_actions(self.state)

        return self.state

    def render(self):
        super().render()
        print('words changed: ', (np.array(self.init_sentence.split()) != np.array(self.state.split())).sum())

    def r(self, a):
        return super().r(a)

    def get_legal_moves(self):
        return super().get_legal_moves()

    def delta(self, s, a):
        # replace with synonym
        if self.ppl_diff:
            return replace_with_synonym_perplexity(s, a, self.sess, lm=self.lm, tokeniser=self.tokenizer)
        # return replace_with_synonym(s, a, self.sess)
        return replace_with_synonym_greedy(s, a, self.text_model, self.sess)

    def step(self, a):
        # if agent chooses illegal move - negative reward , no change in state
        if a not in self.legal_moves:
            self.turn += 1
            return self.state, -0.001, self.turn >= self.max_turns, self.state

        self.legal_moves.remove(a)
        r, done, new_s = self.r(a)
        self.turn += 1

        # limit number of turns in round
        done = (done or (self.turn >= self.max_turns))

        self.score += r
        prev_state = copy(self.state)
        self.state = new_s

        # whether or not round ended
        done = (done or (len(self.legal_moves) == 0))

        # actions with no affect are also considered bad
        if prev_state == self.state:
            return self.state, r, done, self.state

        return self.state, r, done, prev_state

    def get_embedded_state(self, state):
        return self.text_model.embed(state)
