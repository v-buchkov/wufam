from __future__ import annotations

from abc import ABC

from wufam.estimation.base_estimator import BaseEstimator


class BaseMuEstimator(BaseEstimator, ABC):
    def __init__(self) -> None:
        super().__init__()
