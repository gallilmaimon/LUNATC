import numpy as np
from src.Attacks.Agents.Normalizers.Normalizer import Normalizer
from src.Attacks.Agents.Normalizers.OfflineNormalizer import OfflineNormalizer
from src.Attacks.Agents.Normalizers.OnlineNormalizer import OnlineNormalizer


def get_normaliser(num_inputs: int, norm_rounds, states: np.array = None, norm_lock=None, device='numpy') -> Normalizer:
    if states is None:
        return OnlineNormalizer(num_inputs, norm_rounds, norm_lock)
    return OfflineNormalizer(states, device=device)
