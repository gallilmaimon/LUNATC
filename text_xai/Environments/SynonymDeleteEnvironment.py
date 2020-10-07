# imports
import numpy as np
from copy import deepcopy as copy
from text_xai.Environments.utils.action_utils import get_similarity, replace_with_synonym, possible_actions, remove_word

from text_xai.Environments.Environment import Environment


class SynonymDeleteEnvironment(Environment):
    """
    an Environment for A3C agent which also adds to state representation the initial/target class
    to help the model generalise between different sentences.
    an environment which contains only synonyms as possible action
    """
    def __init__(self, max_sent_len, sent_list, sess, init_sentence=None, text_model=None, max_turns=30):
        super().__init__(max_sent_len, sent_list, sess, init_sentence, text_model, max_turns)

    def reset(self, init_sentence=None):
        embedded_state = super().reset(init_sentence)
        self.legal_moves = possible_actions(self.state) + list(range(self.max_sent_len, 2*self.max_sent_len))
        return embedded_state

    def render(self):
        super().render()

    def r(self, a):
        return super().r(a)

    def get_legal_moves(self):
        return super().get_legal_moves()

    def delta(self, s, a):
        # replace with synonym
        if 0 <= a < self.max_sent_len:
            return replace_with_synonym(s, a, self.sess)
        # remove word
        elif a >= self.max_sent_len:
            return remove_word(s, a-self.max_sent_len)

    def step(self, a):
        # if agent chooses illegal move - negative reward , no change in state
        embedded_state = self.get_embedded_state(self.state)
        if a not in self.legal_moves:
            self.turn += 1
            return embedded_state, -0.001, self.turn >= self.max_turns, embedded_state

        r, done, new_s = self.r(a)
        self.legal_moves = possible_actions(new_s) + list(range(self.max_sent_len, 2 * self.max_sent_len))
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
            return embedded_state, r, done, embedded_state

        return embedded_state, r, done, self.get_embedded_state(prev_state)

    def get_embedded_state(self, state):
        return super().get_embedded_state(state)
