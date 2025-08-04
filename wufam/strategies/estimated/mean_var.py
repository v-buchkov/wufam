from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wufam.strategies.optimization_data import PredictionData, TrainingData

import pandas as pd

from wufam.config.trading_config import TradingConfig
from wufam.estimation.mean.base_mu_estimator import BaseMuEstimator
from wufam.estimation.covariance.base_cov_estimator import BaseCovEstimator
from wufam.optimization.constraints import Constraints
from wufam.optimization.optimization import MeanVarianceOptimizer
from wufam.strategies.estimated.base_estimated_strategy import BaseEstimatedStrategy


class MeanVariance(BaseEstimatedStrategy):
    PERCENTAGE_VALID_POINTS = 1.0

    def __init__(
        self,
        mu_estimator: BaseMuEstimator,
        cov_estimator: BaseCovEstimator,
        trading_config: TradingConfig,
        window_size: int | None = None,
    ) -> None:
        super().__init__(
            trading_config=trading_config,
            window_size=window_size,
        )

        self.mu_estimator = mu_estimator
        self.cov_estimator = cov_estimator

    def _fit_estimator(self, training_data: TrainingData) -> None:
        self.mu_estimator.fit(training_data)
        self.cov_estimator.fit(training_data)

    def _optimize(self, prediction_data: PredictionData) -> pd.Series[float]:
        mu = self.mu_estimator.predict(prediction_data)
        covmat = self.cov_estimator.predict(prediction_data)
        constraints = Constraints(ids=self.available_assets)

        if (
            self.trading_config.min_exposure is None
            or self.trading_config.max_exposure is None
        ):
            constr_type = "Unbounded"
        elif self.trading_config.min_exposure >= 0:
            constr_type = "LongOnly"
        else:
            constr_type = "LongShort"

        constraints.add_box(
            box_type=constr_type,
            lower=self.trading_config.min_exposure,
            upper=self.trading_config.max_exposure,
        )
        constraints.add_budget(rhs=self.trading_config.total_exposure, sense="=")

        self.var_min = MeanVarianceOptimizer(constraints=constraints)

        self.var_min.set_objective(mu=mu, covmat=covmat)
        self.var_min.solve()

        return pd.Series(self.var_min.results["weights"])
