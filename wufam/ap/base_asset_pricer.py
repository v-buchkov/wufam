from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseAssetPricer(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def fit(self, test_assets_xs_r: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def predict(self, factors: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def score_r2(self, test_assets_xs_r: pd.DataFrame, factors: pd.DataFrame) -> float:
        pred_xs_r = self.predict(factors)

