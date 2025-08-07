from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wufam.strategies.optimization_data import PredictionData, TrainingData

from abc import ABC, abstractmethod

import pandas as pd

from wufam.config.trading_config import TradingConfig
from wufam.strategies.base_strategy import BaseStrategy


class BaseEstimatedStrategy(BaseStrategy, ABC):
    PERCENTAGE_VALID_POINTS = 1.0

    def __init__(
        self,
        trading_config: TradingConfig,
        window_size: int | None = None,
    ) -> None:
        super().__init__()

        self.trading_config = trading_config
        self.window_size = window_size

    def _fit(self, training_data: TrainingData) -> None:
        ret = training_data.simple_excess_returns[self.available_assets]

        start_date = (
            ret.index[-1] - pd.Timedelta(days=self.window_size)
            if self.window_size is not None
            else ret.index[0]
        )
        ret = ret.loc[start_date:]

        n_valid_points = (~ret.isna()).sum(axis=0) / len(ret)
        valid_stocks = list(
            n_valid_points[n_valid_points >= self.PERCENTAGE_VALID_POINTS].index
        )

        self.available_assets = valid_stocks

        training_data.simple_excess_returns = training_data.simple_excess_returns[
            self.available_assets
        ]
        training_data.simple_excess_returns = training_data.simple_excess_returns.loc[
            start_date:
        ]

        training_data.log_excess_returns = (
            training_data.log_excess_returns.loc[start_date:, self.available_assets]
            if training_data.log_excess_returns is not None
            else None
        )
        training_data.factors = (
            training_data.factors.loc[start_date:]
            if training_data.factors is not None
            else None
        )

        self._fit_estimator(training_data=training_data)

    @abstractmethod
    def _fit_estimator(self, training_data: TrainingData) -> None:
        raise NotImplementedError

    @abstractmethod
    def _optimize(self, prediction_data: PredictionData) -> pd.Series[float]:
        raise NotImplementedError

    def _get_weights(
        self, prediction_data: PredictionData, weights_: pd.DataFrame
    ) -> pd.DataFrame:
        weights = self._optimize(prediction_data=prediction_data)

        weights_.loc[:, weights.index] = weights.to_numpy()

        # (!!!) Please, use only excess returns to apply this scaling correctly
        return weights_
