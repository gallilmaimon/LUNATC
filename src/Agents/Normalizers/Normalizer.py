# abstract class
from abc import ABC, abstractmethod


class Normalizer(ABC):

    @abstractmethod
    def __init__(self):
        print("init normaliser")
        pass

    @abstractmethod
    def observe(self, x):
        pass

    @abstractmethod
    def normalize(self, inputs):
        return inputs
