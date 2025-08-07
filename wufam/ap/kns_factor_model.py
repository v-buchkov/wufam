from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
import statsmodels.api as sm

from wufam.ap.base_asset_pricer import BaseAssetPricer
from wufam.features.ols_betas import get_exposures

warnings.filterwarnings("ignore")


class KNSFactorModel(BaseAssetPricer):
    def __init__(self, l1_penalty: float, l2_penalty: float) -> None:
        super().__init__()

        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty

        self._means = None
        self._vols = None

        self._pca = None
        self._w_mean_var = None
        self._factor_means = None
        self._selection = None

        self._betas = None

    def fit(self, test_assets_xs_r: pd.DataFrame) -> None:
        self._means = test_assets_xs_r.mean()
        self._vols = test_assets_xs_r.std()

        test_assets_xs_r_ss = (test_assets_xs_r - self._means) / (self._vols + 1e-10)

        self._pca = PCA(n_components=min(test_assets_xs_r_ss.shape))
        factors = self._pca.fit_transform(test_assets_xs_r_ss)

        factor_names = [f"Factor_{k}" for k in range(1, factors.shape[1] + 1)]
        factors = pd.DataFrame(
            factors, index=test_assets_xs_r.index, columns=factor_names
        )

        _, exposures = get_exposures(
            factors=factors,
            targets=test_assets_xs_r,
            with_const=True,
        )
        exposures = exposures.astype(float)
        factor_means = self._estimate_risk_premia(
            exposures=exposures,
            test_assets_xs_r=test_assets_xs_r,
        )

        signs = np.sign(factor_means)
        factors = signs * factors
        factor_means = signs * factor_means
        exposures = exposures.mul(signs, axis=1)

        self._factor_means = factor_means

        # ranked_factors = factors[:, np.argsort(factors_means)[::-1]]

        factor_vars = factors.var(axis=0)

        self._w_mean_var = (factor_means - self.l1_penalty) / (
            factor_vars + self.l2_penalty
        )
        self._w_mean_var = np.where(
            factor_means >= self.l1_penalty, self._w_mean_var, 0
        )

        self._selection = factors.columns[factor_means >= self.l1_penalty]
        self._factor_means = factor_means[self._selection]

        self._betas = exposures

    def _get_deviations(
        self, test_assets_xs_r: pd.DataFrame, factors: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series]:
        ts_average = test_assets_xs_r.mean(axis=0)

        model_deviation = ts_average - self.predict()
        baseline_deviation = ts_average - ts_average.mean() * np.ones(len(ts_average))

        return model_deviation, baseline_deviation

    @staticmethod
    def _estimate_risk_premia(
        exposures: pd.DataFrame, test_assets_xs_r: pd.DataFrame
    ) -> pd.Series:
        y = test_assets_xs_r.mean(axis=0).to_frame("cs_mean")
        X = exposures

        results = sm.OLS(y, X).fit()
        return results.params

    def predict(self) -> pd.DataFrame:
        return self._betas[self._selection] @ self._factor_means.T
