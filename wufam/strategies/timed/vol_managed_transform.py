from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wufam.strategies.optimization_data import PredictionData, TrainingData

import numpy as np
import pandas as pd

from wufam.config.trading_config import TradingConfig
from wufam.strategies.timed.vol_managed_strategy import VolManagedStrategy


class VolManagedTransform(VolManagedStrategy):
    def __init__(
        self,
        trading_config: TradingConfig,
        vol_window: int,
    ) -> None:
        super().__init__(
            trading_config=trading_config,
            vol_window=vol_window,
        )

        self.vol_window = vol_window

        self._estimation_date = None
        self._current_vols = None
        self._rebal_date = None

        self._vols_history = []

    def _combine_strategies(self, weights: pd.Series) -> pd.Series[float]:
        return weights / len(weights)
