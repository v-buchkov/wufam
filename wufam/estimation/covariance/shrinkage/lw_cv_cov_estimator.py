from __future__ import annotations

from typing import TYPE_CHECKING

import warnings

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

from sklearn.model_selection import BaseCrossValidator
from sklearn.covariance import ShrunkCovariance, LedoitWolf
from sklearn.model_selection import TimeSeriesSplit

from wufam.strategies.optimization_data import PredictionData, TrainingData
from wufam.estimation.covariance.base_cov_estimator import BaseCovEstimator

warnings.filterwarnings("ignore")


class LedoitWolfCVCovEstimator(BaseCovEstimator):
    def __init__(
        self,
        alphas: list[float] | np.ndarray[float] | None = None,
        cv: BaseCrossValidator = TimeSeriesSplit(n_splits=5),
    ) -> None:
        super().__init__()

        self.alphas = alphas
        self.cv = cv

        self._fitted_cov = None
        self.best_alpha = None

        self.history_alphas = []

    def _fit(self, training_data: TrainingData) -> None:
        ret = training_data.simple_excess_returns

        if self.alphas is not None:
            self.best_alpha = self._find_cv_shrinkage(ret)
            self.history_alphas.append(self.best_alpha)

            lw = ShrunkCovariance(shrinkage=self.best_alpha)
            lw.fit(ret)
            self._fitted_cov = lw.covariance_
        else:
            lw = LedoitWolf(store_precision=False)
            lw.fit(ret)
            self._fitted_cov = lw.covariance_
            self.best_alpha = lw.shrinkage_

    def _find_cv_shrinkage(self, ret: pd.DataFrame) -> float:
        alphas = self.alphas
        cv = self.cv
        best_alpha = 0
        best_score = np.inf

        for alpha in alphas:
            scores = []
            for train, test in cv.split(ret):
                lw = ShrunkCovariance(shrinkage=alpha)
                lw.fit(ret.iloc[train])
                scores.append(self._evaluate_covariance(lw.covariance_, ret.iloc[test]))

            score = np.mean(scores)
            if score < best_score:
                best_alpha = alpha
                best_score = score

        return best_alpha

    @staticmethod
    def _evaluate_covariance(covmat: pd.DataFrame, rets: pd.DataFrame) -> float:
        # N x N
        covmat_inv = np.linalg.inv(covmat)
        # N x 1
        ones = np.ones((rets.shape[1], 1))

        # N x 1
        w_opt = covmat_inv @ ones
        w_opt = w_opt / np.sum(w_opt)

        return w_opt.T @ rets.cov() @ w_opt

    def _predict(self, prediction_data: PredictionData) -> pd.DataFrame:
        return self._fitted_cov
