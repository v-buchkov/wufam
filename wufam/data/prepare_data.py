from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_factors(
    filename: Path,
    start_date: pd.Timestamp | str | None = None,
    end_date: pd.Timestamp | str | None = None,
) -> pd.Series:
    factors = pd.read_csv(
        filename, skiprows=4, skipfooter=3, index_col=0, engine="python"
    )
    factors.index = pd.to_datetime(factors.index, format="%Y%m%d")
    return factors.loc[start_date:end_date] / 100


def read_kf_portfolios(
    filename: Path,
    start_date: pd.Timestamp | str | None = None,
    end_date: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    portfolios_df = pd.read_csv(
        filename,
        skiprows=26_045,
        skipfooter=104_126 - 52_070,
        index_col=0,
        engine="python",
    )
    portfolios_df.index = pd.to_datetime(portfolios_df.index, format="%Y%m%d")
    return portfolios_df.loc[start_date:end_date] / 100


def read_kf_data(
    portfolios_filename: Path,
    factors_filename: Path,
    start_date: pd.Timestamp | str | None = None,
    end_date: pd.Timestamp | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    portfolios = read_kf_portfolios(portfolios_filename, start_date, end_date)
    factors = read_factors(factors_filename, start_date, end_date)

    factors = portfolios.merge(factors, left_index=True, right_index=True, how="left")
    rf = factors["RF"]
    factors = factors.drop(columns=["RF"])

    return portfolios, factors, rf
