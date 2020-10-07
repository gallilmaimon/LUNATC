import torch
from src.Agents.Normalizers.Normalizer import Normalizer


class OnlineNormalizer(Normalizer):
    def __init__(self, num_inputs, norm_rounds, lock):
        super().__init__()
        self.n = torch.zeros(1, num_inputs)
        self.mean = torch.zeros(1, num_inputs)
        self.mean_diff = torch.zeros(1, num_inputs)
        self.var = torch.ones(1, num_inputs)
        self.norm_rounds = norm_rounds
        self.lock = lock

    def observe(self, x):
        self.n += 1.
        if self.n[0][0] < self.norm_rounds:
            self.lock.acquire()
            last_mean = self.mean.clone()
            self.mean += (x-self.mean)/self.n
            self.mean_diff += (x-last_mean)*(x-self.mean)
            self.var = torch.clamp(self.mean_diff/self.n, min=1e-8)
            self.lock.release()

    def normalize(self, inputs):
        self.lock.acquire()
        obs_std = torch.sqrt(self.var)
        normalised = (inputs - self.mean)/obs_std
        self.lock.release()
        return normalised
