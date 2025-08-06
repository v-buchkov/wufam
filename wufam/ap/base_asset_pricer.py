from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseAssetPricer(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def fit(
        self, test_assets_xs_r: pd.DataFrame, factors: pd.DataFrame
    ) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def predict(self, factors: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def _get_deviations(
        self, test_assets_xs_r: pd.DataFrame, factors: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series]:
        pred_xs_r = self.predict(factors)

        ts_average = test_assets_xs_r.mean(axis=0)

        model_deviation = ts_average - pred_xs_r.mean(axis=1)
        baseline_deviation = ts_average - ts_average.mean() * np.ones(len(ts_average))

        return model_deviation, baseline_deviation

    def r2_score(self, test_assets_xs_r: pd.DataFrame, factors: pd.DataFrame) -> float:
        model_deviation, baseline_deviation = self._get_deviations(
            test_assets_xs_r, factors
        )

        mse_model = model_deviation.T @ model_deviation
        mse_baseline = baseline_deviation.T @ baseline_deviation

        return 1 - mse_model / mse_baseline

    def r2_gls_score(
        self, test_assets_xs_r: pd.DataFrame, factors: pd.DataFrame
    ) -> float:
        model_deviation, baseline_deviation = self._get_deviations(
            test_assets_xs_r, factors
        )

        var_r_inv = np.linalg.inv(test_assets_xs_r.cov())

        mse_model = model_deviation.T @ var_r_inv @ model_deviation
        mse_baseline = baseline_deviation.T @ var_r_inv @ baseline_deviation

        return 1 - mse_model / mse_baseline

    def implied_sharpe_ratio(
        self, test_assets_xs_r: pd.DataFrame, factors: pd.DataFrame
    ) -> float: ...

    def rmse_score(
        self, test_assets_xs_r: pd.DataFrame, factors: pd.DataFrame
    ) -> float:
        model_deviation, baseline_deviation = self._get_deviations(
            test_assets_xs_r, factors
        )

        return np.sqrt(model_deviation.T @ model_deviation / len(model_deviation))
