from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from wufam.strategies.optimization_data import TrainingData, PredictionData


class BaseEstimator(ABC):
    def __init__(self) -> None:
        super().__init__()

        self._fitted = False
        self._available_assets = None

    def fit(self, training_data: TrainingData) -> None:
        self._fit(training_data=training_data)
        self._fitted = True

    def predict(self, prediction_data: PredictionData) -> pd.DataFrame:
        assert self._fitted, "Model is not fitted yet! Call fit() first."

        return self._predict(prediction_data=prediction_data)

    def __call__(self, prediction_data: PredictionData) -> pd.DataFrame:
        return self.predict(prediction_data=prediction_data)

    @abstractmethod
    def _fit(self, training_data: TrainingData) -> None:
        raise NotImplementedError

    @abstractmethod
    def _predict(self, prediction_data: PredictionData) -> pd.Series | pd.DataFrame:
        raise NotImplementedError

    @property
    def available_assets(self) -> list[str]:
        return self._available_assets

    @available_assets.setter
    def available_assets(self, available_assets: list[str]) -> None:
        self._available_assets = available_assets
