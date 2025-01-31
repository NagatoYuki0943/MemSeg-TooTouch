from abc import ABC, abstractmethod
import numpy as np

class Inference(ABC):

    @abstractmethod
    def infer(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError
