from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import f

from wufam.ap.base_asset_pricer import BaseAssetPricer
from wufam.features.ols_betas import get_exposures


class UncondFactorModel(BaseAssetPricer):
    def __init__(self, store_resids: bool = False) -> None:
        super().__init__()

        self.store_resids = store_resids

        self._alphas = None
        self._betas = None
        self._resids = None

        self._grs_stat = None
        self._rv_f = None

        self._pval = None

    def fit(self, rets_df: pd.DataFrame, factors_df: pd.DataFrame) -> None:
        alphas, betas, resids = get_exposures(
            factors=factors_df, targets=rets_df, return_residuals=True
        )

        resid_cov = resids.cov()
        resid_cov_inv = np.linalg.inv(resid_cov)

        df_1 = rets_df.shape[1]
        df_2 = rets_df.shape[0] - rets_df.shape[1] - 1

        const_adj = (
            1 + (factors_df.mean(axis=0).mean() / factors_df.std(axis=0).mean()) ** 2
        )
        const_adj = 1 / const_adj

        self._grs_stat = df_2 / df_1 * const_adj * (alphas.T @ resid_cov_inv @ alphas)

        self._rv_f = f(df_1, df_2)
        self._pval = self._rv_f.sf(self._grs_stat)

        self._alphas = alphas
        self._betas = betas
        if self.store_resids:
            self._resids = resids

    def predict(self, factors_df: pd.DataFrame) -> pd.DataFrame:
        return self._betas @ factors_df.T

    @property
    def grs_stat(self) -> float:
        return self._grs_stat

    def get_critical_value(self, significance: float = 0.05) -> float:
        return self._rv_f.ppf(1 - significance)

    @property
    def critical_value(self) -> float:
        return self.get_critical_value()

    @property
    def p_value(self) -> float:
        return self._pval
