# abstract class
from abc import ABC, abstractmethod


class TextModel(ABC):

    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def test(self, X, y):
        pass

    @abstractmethod
    def embed(self, X):
        pass

    @abstractmethod
    def predict_proba(self, X):
        pass

    @abstractmethod
    def predict(self, X):
        pass
