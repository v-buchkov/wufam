from __future__ import annotations

from pathlib import Path
from enum import Enum

import pandas as pd


class Weighting(Enum):
    EW = "equally_weighted"
    VW = "value_weighted"


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
    weighting: Weighting = Weighting.EW,
) -> pd.DataFrame:
    if weighting == Weighting.EW:
        start_row = 26_045
        end_row = 52_070
    else:
        start_row = 18
        end_row = 26_042

    portfolios_df = pd.read_csv(
        filename,
        skiprows=start_row,
        skipfooter=104_126 - end_row,
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
    weighting: str = "equally_weighted",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    weighting = Weighting(weighting)

    portfolios_total_r = read_kf_portfolios(
        portfolios_filename, start_date, end_date, weighting
    )
    factors = read_factors(factors_filename, start_date, end_date)

    factors = portfolios_total_r.merge(
        factors, left_index=True, right_index=True, how="left"
    )
    rf = factors["RF"]
    factors = factors.drop(columns=["RF"] + portfolios_total_r.columns.tolist())

    portfolios_xs_r = portfolios_total_r.sub(rf, axis=0)

    return portfolios_total_r, portfolios_xs_r, factors, rf
