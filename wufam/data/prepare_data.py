from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_rf_rate(
    filename: Path,
    start_date: pd.Timestamp | str | None = None,
    end_date: pd.Timestamp | str | None = None,
) -> pd.Series:
    rf = pd.read_csv(filename, skiprows=4, skipfooter=3, index_col=0, engine="python")[
        "RF"
    ]
    rf.index = pd.to_datetime(rf.index, format="%Y%m%d")
    return rf.loc[start_date:end_date] / 100


def read_kf_factors(
    filename: Path,
    start_date: pd.Timestamp | str | None = None,
    end_date: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    factors_df = pd.read_csv(
        filename,
        skiprows=26_045,
        skipfooter=104_126 - 52_070,
        index_col=0,
        engine="python",
    )
    factors_df.index = pd.to_datetime(factors_df.index, format="%Y%m%d")
    return factors_df.loc[start_date:end_date] / 100


def read_kf_data(
    factors_filename: Path,
    rf_filename: Path,
    start_date: pd.Timestamp | str | None = None,
    end_date: pd.Timestamp | str | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    factors = read_kf_factors(factors_filename, start_date, end_date)
    rf = read_rf_rate(rf_filename, start_date, end_date)

    rf = factors.merge(rf, left_index=True, right_index=True, how="left")["RF"]

    return factors, rf
