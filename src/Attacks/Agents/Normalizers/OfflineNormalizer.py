import torch
from src.Attacks.Agents.Normalizers.Normalizer import Normalizer


class OfflineNormalizer(Normalizer):
    def __init__(self, states, device='numpy'):
        super().__init__()
        self.mean = torch.Tensor(states.mean(axis=0))
        self.var = torch.Tensor(states.var(axis=0))
        if device != 'numpy':
            self.mean = self.mean.to(device)
            self.var = self.var.to(device)

    def observe(self, x):
        pass

    def normalize(self, inputs):
        obs_std = torch.clamp(torch.sqrt(self.var), min=1e-8)
        return (inputs - self.mean)/obs_std
