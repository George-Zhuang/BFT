from torch.nn import Module
from abc import ABCMeta, abstractmethod

class BaseExp(metaclass=ABCMeta):
    """Basic class for any experiment."""
    def __init__(self):
        pass

    @abstractmethod
    def get_det_model(self) -> Module:
        pass

    @abstractmethod
    def get_tracker(self) -> Module:
        pass
