from __future__ import annotations

import numpy as np
import pandas as pd

from tqdm import tqdm

from wufam.strategies.base_strategy import BaseStrategy
from wufam.strategies.optimization_data import PredictionData, TrainingData


def run_rolling_features_backtest(
    strategy: BaseStrategy,
    excess_returns: pd.DataFrame,
    factors: pd.DataFrame,
    cross_sectional_features: pd.DataFrame,
    targets: pd.Series,
    rf: pd.Series,
    freq: str | None = None,
    trading_lag: int = 1,
    window_size: int | None = None,
    return_weights: bool = False,
) -> tuple[pd.DataFrame, pd.Series] | tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    schedule = generate_rebal_schedule(
        dates=excess_returns.index, ret=excess_returns, freq=freq
    )[4:]

    weights = get_rolling_weights(
        strategy,
        excess_returns,
        factors,
        cross_sectional_features,
        targets,
        schedule,
        trading_lag,
        window_size,
    ).astype(float)

    total_returns = excess_returns.add(rf, axis=0)
    me_weights, strategy_total_r = float_weights(
        total_returns=total_returns, weights=weights, rf=rf
    )

    turnover = calc_turnover(weights=weights, month_end_weights=me_weights)

    if return_weights:
        return strategy_total_r, turnover, me_weights

    return strategy_total_r, turnover


def calc_turnover(weights: pd.DataFrame, month_end_weights: pd.DataFrame) -> pd.Series:
    return month_end_weights.diff().dropna().loc[weights.index[1:]].abs().sum(axis=1)


def generate_rebal_schedule(
    dates: pd.Index, ret: pd.DataFrame, freq: str | None
) -> pd.DatetimeIndex:
    date_index = dates

    if freq == "each":
        return date_index[1:]

    if freq is None:
        schedule = date_index
    else:
        schedule = ret.groupby(date_index.to_period(freq.rstrip("E"))).tail(1).index

    if schedule[-1] == date_index[-1]:
        schedule = schedule[:-1]

    if freq is None:
        schedule = schedule[:1]

    return schedule


def get_rolling_weights(
    strategy: BaseStrategy,
    excess_returns: pd.DataFrame,
    factors: pd.DataFrame,
    cross_sectional_features: pd.DataFrame,
    targets: pd.Series,
    schedule: pd.DatetimeIndex,
    trading_lag: int = 1,
    window_size: int | None = None,
) -> pd.DataFrame:
    weights = pd.DataFrame(index=schedule, columns=excess_returns.columns)
    for date in tqdm(schedule, desc="Optimizing Strategy"):
        if window_size is not None:
            start_date = date - pd.Timedelta(days=window_size)
        else:
            start_date = None

        cs_features = cross_sectional_features.loc[
            start_date : date - pd.Timedelta(trading_lag), :
        ]
        tgts = targets.loc[start_date : date - pd.Timedelta(trading_lag)]
        dates = cs_features.index.get_level_values(0).unique()

        training_data = TrainingData(
            simple_excess_returns=excess_returns.loc[
                start_date : date - pd.Timedelta(trading_lag)
            ],
            factors=factors.loc[start_date : date - pd.Timedelta(trading_lag)],
            cross_sectional_features=cs_features.loc[dates[:-1], :],
            targets=tgts.loc[dates[1:], :],
        )

        strategy.fit(training_data)

        weights.loc[date, :] = strategy(
            PredictionData(
                cross_sectional_features=cs_features.loc[dates[-1:], :],
            )
        ).to_numpy()

    return weights


def float_weights(
    total_returns: pd.DataFrame,
    weights: pd.DataFrame,
    rf: pd.Series,
    add_total_r: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    total_r = pd.DataFrame(
        index=total_returns.index, columns=["total_r"], dtype=np.float64
    )
    level_data = pd.DataFrame(
        index=total_returns.index, columns=["level_data"], dtype=np.float64
    )
    float_weights = pd.DataFrame(
        index=total_returns.index, columns=[*total_returns.columns.tolist()]
    )

    total_returns = (
        pd.concat([total_returns, rf], axis=1)
        if add_total_r is None
        else pd.concat([total_returns, add_total_r, rf], axis=1)
    )
    n_auxilary_cols = 1 if add_total_r is None else 2
    weights = weights.copy()
    if add_total_r is not None:
        weights["add"] = 1
    weights["rf"] = (1 - weights.sum(axis=1)).round(5)

    last_rebal_date = weights.index[0]

    total_r = total_r.loc[last_rebal_date:]
    float_weights = float_weights.loc[last_rebal_date:]
    level_data = level_data.loc[last_rebal_date:]

    total_r.loc[last_rebal_date] = np.float64(0.0)
    float_weights.loc[last_rebal_date] = np.float64(0.0)
    for rebal in [*weights.index[1:].tolist(), None]:
        w0 = weights.loc[last_rebal_date]
        start_date = last_rebal_date
        end_date = rebal if rebal else None

        sample_r = total_returns.loc[start_date:end_date].copy().fillna(0)
        r_mat = 1 + sample_r
        r_mat.iloc[0] = w0.fillna(0)
        float_w = r_mat.cumprod(axis=0).fillna(0)

        level = float_w.sum(axis=1)

        ret_tmp = level.pct_change(1).iloc[1:]

        total_r.loc[ret_tmp.index] = ret_tmp.to_frame()
        normalized = float_w.iloc[:, :-n_auxilary_cols].div(level, axis=0)
        float_weights.loc[sample_r.index, :] = normalized
        level_data.loc[level.index] = level.to_frame()

        last_rebal_date = rebal

    return float_weights, total_r
