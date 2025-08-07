from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wufam.strategies.optimization_data import PredictionData, TrainingData

import numpy as np
import pandas as pd

from wufam.config.trading_config import TradingConfig
from wufam.strategies.optimized.base_estimated_strategy import BaseEstimatedStrategy


class VolManagedStrategy(BaseEstimatedStrategy):
    def __init__(
        self,
        trading_config: TradingConfig,
        vol_window: int,
    ) -> None:
        super().__init__(
            trading_config=trading_config,
            window_size=None,
        )

        self.vol_window = vol_window

        self._estimation_date = None
        self._current_vols = None
        self._rebal_date = None

        self._vols_history = []

    def _fit_estimator(self, training_data: TrainingData) -> None:
        xs_r = training_data.simple_excess_returns
        last_date = xs_r.index[-1] - pd.Timedelta(days=self.vol_window)
        self._rebal_date = xs_r.index[-1]
        self._current_vols = xs_r.loc[last_date:].var(axis=0).to_numpy()
        self._vols_history.append(self._current_vols)

    def _optimize(self, prediction_data: PredictionData) -> pd.DataFrame:
        weights = np.array(self._vols_history[:-1]).mean(axis=0) / self._current_vols if len(self._vols_history) > 3 else np.ones_like(self._current_vols).reshape(-1, 1)
        weights = np.clip(weights, self.trading_config.min_exposure, self.trading_config.max_exposure)
        weights = self._combine_strategies(weights)

        return pd.Series(weights.flatten(), index=self.available_assets)

    def _combine_strategies(self, weights: pd.Series) -> pd.Series[float]:
        return weights / len(weights)
