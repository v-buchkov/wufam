from __future__ import annotations

import numpy as np
import pandas as pd


def calc_sharpe(
    strategy_total_r: pd.DataFrame,
    rf_rate: pd.Series,
    factor_annualize: float | None = None,
) -> float:
    if factor_annualize is None:
        day_diff = strategy_total_r.index.diff().days
        factor_annualize = round(np.nanmean(365 // day_diff))

    n_periods = strategy_total_r.shape[0] / factor_annualize

    final_rf = rf_rate.add(1).prod()
    final_nav = strategy_total_r.add(1).prod()

    strat_mean = final_nav ** (1 / n_periods) - 1
    rf_mean = final_rf ** (1 / n_periods) - 1

    strat_vol = strategy_total_r.std() * np.sqrt(factor_annualize)

    sr_annual = (strat_mean - rf_mean) / strat_vol

    # TODO(@V): Add serial correlation adjustment
    return (
        sr_annual.iloc[0].item()
        if isinstance(sr_annual, pd.DataFrame)
        else sr_annual.item()
    )


def calc_sharpe_from_weights(
    weights: pd.DataFrame,
    excess_ret: pd.DataFrame,
    rf_rate: pd.Series,
    factor_annualize: float | None = None,
) -> float:
    if factor_annualize is None:
        day_diff = excess_ret.index.diff().days
        factor_annualize = round(np.nanmean(365 // day_diff))

    n_periods = excess_ret.shape[0] / factor_annualize

    final_rf = rf_rate.add(1).prod()
    final_total_ret = excess_ret.add(rf_rate, axis=0).add(1).prod(axis=0)
    final_excess_ret = final_total_ret - final_rf

    final_nav = weights @ final_excess_ret + final_rf + 1

    strat_mean = final_nav ** (1 / n_periods) - 1
    rf_mean = final_rf ** (1 / n_periods) - 1

    strat_var = weights @ excess_ret.cov() @ weights.T
    strat_vol = np.sqrt(strat_var * factor_annualize)

    sr_annual = (strat_mean - rf_mean) / strat_vol

    # TODO(@V): Add serial correlation adjustment
    return (
        sr_annual.iloc[0].item()
        if isinstance(sr_annual, pd.DataFrame)
        else sr_annual.item()
    )
