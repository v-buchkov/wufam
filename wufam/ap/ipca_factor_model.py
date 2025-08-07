from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ipca import InstrumentedPCA

from wufam.ap.base_asset_pricer import BaseAssetPricer

warnings.filterwarnings("ignore")


class IPCAFactorModel(BaseAssetPricer):
    def __init__(self, n_factors: int = 4, fit_alpha: bool = True) -> None:
        super().__init__()

        self.n_factors = n_factors
        self.fit_alpha = fit_alpha

        self.ipca = None
        self.Gamma = None
        self.Factors = None

    def fit(self, test_assets_xs_r: pd.Series, ranks: pd.DataFrame) -> None:
        self.ipca = InstrumentedPCA(
            n_factors=self.n_factors,
            intercept=self.fit_alpha,
            max_iter=10_000,
        )
        self.ipca = self.ipca.fit(X=ranks, y=test_assets_xs_r)
        self.Gamma, self.Factors = self.ipca.get_factors(label_ind=True)

    def predict(self, ranks: pd.DataFrame) -> pd.DataFrame:
        pred = self.ipca.predict(X=ranks)
        pred = pd.DataFrame(pred, index=ranks.index, columns=["pred"])
        return pd.pivot_table(pred, index="date", columns="portfolio", values="pred")

    def _get_deviations(
        self, test_assets_xs_r: pd.DataFrame, ranks: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series]:
        pred_xs_r = self.predict(ranks)

        ts_average = test_assets_xs_r.mean(axis=0)

        model_deviation = ts_average - pred_xs_r.mean(axis=0)
        baseline_deviation = ts_average - ts_average.mean() * np.ones(len(ts_average))

        return model_deviation, baseline_deviation

    def r2_score(self, test_assets_xs_r: pd.DataFrame, ranks: pd.DataFrame) -> float:
        model_deviation, baseline_deviation = self._get_deviations(
            test_assets_xs_r, ranks
        )

        mse_model = model_deviation.T @ model_deviation
        mse_baseline = baseline_deviation.T @ baseline_deviation

        return 1 - mse_model / mse_baseline

    def r2_gls_score(
        self, test_assets_xs_r: pd.DataFrame, ranks: pd.DataFrame
    ) -> float:
        model_deviation, baseline_deviation = self._get_deviations(
            test_assets_xs_r, ranks
        )

        var_r_inv = np.linalg.inv(test_assets_xs_r.cov())

        mse_model = model_deviation.T @ var_r_inv @ model_deviation
        mse_baseline = baseline_deviation.T @ var_r_inv @ baseline_deviation

        return 1 - mse_model / mse_baseline

    def get_mv_weights(
        self, test_assets_xs_r: pd.DataFrame, ranks: pd.DataFrame
    ) -> float:
        Z_t = ranks
        Gamma_b = self.Gamma.iloc[:, :-1] if self.fit_alpha else self.Gamma
        var = Gamma_b.T @ Z_t.T @ Z_t @ Gamma_b
        var_inv = np.linalg.inv(var)

        weights = (var_inv @ Gamma_b.T @ Z_t.T).T

        factors_mu = self.Factors.mean(axis=1)
        factors_cov = self.Factors.T.cov()

        mv_weights = np.linalg.inv(factors_cov) @ factors_mu

        factor_rets = pd.DataFrame(
            index=test_assets_xs_r.index,
            columns=[f"Factor{k}" for k in range(1, self.n_factors + 1)],
        )
        for factor in range(self.n_factors):
            factor_weights = pd.pivot_table(
                weights.iloc[:, factor].to_frame("weights"),
                index="date",
                columns="portfolio",
                values="weights",
            )
            factor_ret = (factor_weights.to_numpy() * test_assets_xs_r.to_numpy()).sum(
                axis=1
            )
            factor_rets[f"Factor{factor + 1}"] = factor_ret

        return factor_rets @ mv_weights

    def rmse_score(self, test_assets_xs_r: pd.DataFrame, ranks: pd.DataFrame) -> float:
        model_deviation, baseline_deviation = self._get_deviations(
            test_assets_xs_r, ranks
        )

        return np.sqrt(model_deviation.T @ model_deviation / len(model_deviation))
