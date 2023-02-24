import numpy as np
from sklearn.preprocessing import StandardScaler

from normalizers.normalizer import Normalizer

class StandardNormalizer(Normalizer):
    """
    Normalizes the input by substracting the mean and dividing by standard devation.
    Uses ``sklearn.preprocessing.StandardScaler`` under the hood.
    """

    def __init__(self):
        self.scaler = StandardScaler()

    def partial_fit(self, array: np.ndarray) -> None:
        self.scaler.partial_fit(self._reshape_for_scaler(array))

    def transform(self, array: np.ndarray) -> np.ndarray:
        return self.scaler.transfrom(self._reshape_for_scaler(array)).reshape(array.shape)

    @staticmethod
    def _reshape_for_scaler(array: np.ndarray):
        # N*T*any just scale any part.
        new_shape = (-1, *array.shape[2:]) if array.ndim > 2 else (-1, 1)
        return array.reshape(new_shape)