from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from tqdm import tqdm


def avg_non_diagonal_elements(corr_matrix):
    non_diag = corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)]
    return np.nanmean(non_diag)


def avg_corr(rolling_window):
    corr_matrix = rolling_window.corr()

    return avg_non_diagonal_elements(corr_matrix)


def _rolling_feature(
    df: pd.DataFrame,
    feature_fn: Callable,
    feature_name: str | None = None,
    verbose: bool = False,
):
    # Initialize a list to store results
    results = []

    # Perform calculation for each rolling window
    for end in tqdm(df.index) if verbose else df.index:
        start = end - pd.DateOffset(months=1)
        rolling_window = df.loc[start:end]

        feature = feature_fn(rolling_window)

        results.append([end, feature])

    # Create a series with the results
    feature_name = "feature" if feature_name is None else feature_name
    rolling_feat = pd.DataFrame(results, columns=["date", feature_name])
    rolling_feat["date"] = pd.to_datetime(rolling_feat["date"])
    rolling_feat = rolling_feat.set_index("date")

    return rolling_feat


def create_dnk_features(
    ret: pd.DataFrame,
    verbose: bool = False,
) -> pd.DataFrame:
    # 1. Avg Corr.
    # Calculate rolling average correlation of non-diagonal elements
    rolling_avg_corr = _rolling_feature(
        ret, avg_corr, "avg_corr", verbose=verbose
    )

    # 2. Average volatility.
    avg_vol = _rolling_feature(
        ret, lambda s: s.std(axis=0).mean(), "avg_vol", verbose=verbose
    )

    # 3. EW Portfolio.
    ew = _rolling_feature(
        ret,
        lambda s: np.prod(1 + np.nanmean(s, axis=1)) - 1,
        "ew",
        verbose=verbose,
    )

    # 4. EW Portfolio Moving Average.
    ewma = []
    for end in tqdm(ew.index) if verbose else ew.index:
        start = end - pd.DateOffset(months=1)

        if end > ew.index[-1]:
            break

        sample = ew.loc[start:end]

        ma = sample.ewm(alpha=0.1).mean().iloc[-1].item()

        ewma.append([end, ma])
    ewma = pd.DataFrame(ewma, columns=["date", "ewma"])
    ewma["date"] = pd.to_datetime(ewma["date"])
    ewma = ewma.set_index("date")

    # 4. Ledoit-Wolf Shrinkage Intensity.
    def get_intensity(s: pd.DataFrame):
        s = s.copy().fillna(0)
        lw_estimator = LedoitWolf()
        lw_estimator.fit(s)
        return lw_estimator.shrinkage_

    lw = _rolling_feature(
        ret,
        lambda s: get_intensity(s),
        "lw_shrinkage",
        verbose=verbose,
    )

    # 5. Momentum
    momentum = _rolling_feature(
        ret,
        lambda s: np.nanmean(np.where(s, s > 0, 1), axis=0).mean(),
        "momentum_feature",
        verbose=verbose,
    )

    # 6. Trace.
    trace = _rolling_feature(
        ret,
        lambda s: np.trace(s.fillna(0).cov()),
        "trace",
        verbose=verbose,
    )

    # 7. Universe Volatility.
    ew_vol = ew.rolling(window=252, min_periods=1).std().fillna(0)

    # Merge all features.
    features = rolling_avg_corr.merge(
        avg_vol, how="inner", left_index=True, right_index=True
    )

    features = features.merge(ewma, how="inner", left_index=True, right_index=True)
    features = features.merge(lw, how="inner", left_index=True, right_index=True)
    features = features.merge(momentum, how="inner", left_index=True, right_index=True)
    features = features.merge(trace, how="inner", left_index=True, right_index=True)
    features = features.merge(
        ew_vol.rename(columns={"ew": "universe_vol"}),
        how="inner",
        left_index=True,
        right_index=True,
    )

    if rolling_avg_corr.shape[0] != features.shape[0]:
        msg = "The dates of created features do not match!"
        raise ValueError(msg)

    return features
