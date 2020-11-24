import random
import numpy as np

from src.Agents.Memory.SumTree import SumTree
from src.Agents.Memory.ReplayMemory import Transition


class PrioritisedMemory:
    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

        self.eps = 0.01
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_increment_per_sampling = 0.001

    def _get_priority(self, error):
        return (np.abs(error) + self.eps) ** self.alpha

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, Transition(*sample))

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def __len__(self):
        return self.tree.n_entries
