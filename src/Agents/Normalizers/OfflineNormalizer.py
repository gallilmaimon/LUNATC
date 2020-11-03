import torch
from src.Agents.Normalizers.Normalizer import Normalizer


class OfflineNormalizer(Normalizer):
    def __init__(self, states):
        super().__init__()
        self.mean = torch.Tensor(states.mean(axis=0)).cuda()
        self.var = torch.Tensor(states.var(axis=0)).cuda()

    def observe(self, x):
        pass

    def normalize(self, inputs):
        obs_std = torch.clamp(torch.sqrt(self.var), min=1e-8)
        return (inputs - self.mean)/obs_std
