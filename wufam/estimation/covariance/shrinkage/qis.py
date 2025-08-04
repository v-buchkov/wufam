from __future__ import annotations

import math
import numpy as np
import pandas as pd

from wufam.estimation.covariance.base_cov_estimator import BaseCovEstimator
from wufam.strategies.optimization_data import PredictionData, TrainingData


# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 15:18:30 2021

@author: Patrick Ledoit
"""

# function sigmahat=QIS(Y,k)
#
# Y (N*p): raw data matrix of N iid observations on p random variables
# sigmahat (p*p): invertible covariance matrix estimator
#
# Implements the quadratic-inverse shrinkage (QIS) estimator
#    This is a nonlinear shrinkage estimator derived under the Frobenius loss
#    and its two cousins, Inverse Stein's loss and Mininum Variance loss
#
# If the second (optional) parameter k is absent, not-a-number, or empty,
# then the algorithm demeans the data by default, and adjusts the effective
# sample size accordingly. If the user inputs k = 0, then no demeaning
# takes place; if (s)he inputs k = 1, then it signifies that the data Y has
# already been demeaned.
#
# This version: 01/2021

###########################################################################
# This file is released under the BSD 2-clause license.


# Copyright (c) 2021, Olivier Ledoit and Michael Wolf
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###########################################################################
def _QIS(Y, shrinkage: float = 1.0, k: int | None = None) -> pd.DataFrame:
    # Pre-Conditions: Y is a valid pd.dataframe and optional arg- k which can be
    #    None, np.nan or int
    # Post-Condition: Sigmahat dataframe is returned

    # Set df dimensions
    N = Y.shape[0]  # num of columns
    p = Y.shape[1]  # num of rows

    # default setting
    if k is None or math.isnan(k):
        Y = Y.sub(Y.mean(axis=0), axis=1)  # demean
        k = 1

    # vars
    n = N - k  # adjust effective sample size
    c = p / n  # concentration ratio

    # Cov df: sample covariance matrix
    sample = pd.DataFrame(np.matmul(Y.T.to_numpy(), Y.to_numpy())) / n
    sample = (sample + sample.T) / 2  # make symmetrical

    # Spectral decomp
    lambda1, u = np.linalg.eig(sample)  # use LAPACK routines
    lambda1 = lambda1.real  # clip imaginary part due to rounding error
    u = u.real  # clip imaginary part for eigenvectors

    lambda1 = lambda1.real.clip(min=0)  # reset negative values to 0
    dfu = pd.DataFrame(u, columns=lambda1)  # create df with column names lambda
    #                                        and values u
    dfu.sort_index(axis=1, inplace=True)  # sort df by column index
    lambda1 = dfu.columns  # recapture sorted lambda

    # COMPUTE Quadratic-Inverse Shrinkage estimator of the covariance matrix
    h = (min(c**2, 1 / c**2) ** 0.35) / p**0.35  # smoothing parameter
    invlambda = (
        1 / lambda1[max(1, p - n + 1) - 1 : p]
    )  # inverse of (non-null) eigenvalues
    dfl = pd.DataFrame()
    dfl["lambda"] = invlambda
    Lj = dfl[np.repeat(dfl.columns.values, min(p, n))]  # like  1/lambda_j
    Lj = pd.DataFrame(Lj.to_numpy())  # Reset column names
    Lj_i = Lj.subtract(Lj.T)  # like (1/lambda_j)-(1/lambda_i)

    theta = (
        Lj.multiply(Lj_i)
        .div(Lj_i.multiply(Lj_i).add(Lj.multiply(Lj) * h**2))
        .mean(axis=0)
    )  # smoothed Stein shrinker
    Htheta = (
        Lj.multiply(Lj * h)
        .div(Lj_i.multiply(Lj_i).add(Lj.multiply(Lj) * h**2))
        .mean(axis=0)
    )  # its conjugate
    Atheta2 = theta**2 + Htheta**2  # its squared amplitude

    if p <= n:  # case where sample covariance matrix is not singular
        delta = 1 / (
            (1 - c) ** 2 * invlambda
            + 2 * c * (1 - c) * invlambda * theta
            + c**2 * invlambda * Atheta2
        )  # optimally shrunk eigenvalues
        delta = delta.to_numpy()
    else:
        delta0 = 1 / ((c - 1) * np.mean(invlambda.to_numpy()))  # shrinkage of null
        #                                                 eigenvalues
        delta = np.repeat(delta0, p - n)
        delta = np.concatenate((delta, 1 / (invlambda * Atheta2)), axis=None)

    deltaQIS = delta * (sum(lambda1) / sum(delta))  # preserve trace

    init_delta = deltaQIS.copy()
    deltaQIS = shrinkage * init_delta + (1 - shrinkage) * lambda1.to_numpy()

    # Apply isotonic regression by (De Nard & Kostovic, 2025)
    smallest = np.min(init_delta[init_delta < lambda1])
    largest = np.max(init_delta[init_delta > lambda1])
    crossing_point = (smallest + largest) / 2

    deltaQIS[-1] = np.maximum(deltaQIS[-1], crossing_point)
    for i in range(0, len(deltaQIS) - 1)[::-1]:
        if init_delta[i] < lambda1[i]:
            deltaQIS[i] = np.maximum(
                np.minimum(deltaQIS[i], deltaQIS[i + 1]), crossing_point
            )
        else:
            deltaQIS[i] = np.minimum(
                np.minimum(deltaQIS[i], deltaQIS[i + 1]), crossing_point
            )

    temp1 = dfu.to_numpy()
    temp2 = np.diag(deltaQIS)
    temp3 = dfu.T.to_numpy().conjugate()
    # reconstruct covariance matrix
    sigmahat = pd.DataFrame(np.matmul(np.matmul(temp1, temp2), temp3))
    return sigmahat


class QISCovEstimator(BaseCovEstimator):
    def __init__(self, shrinkage: float = 1.0) -> None:
        super().__init__()

        self.shrinkage = shrinkage

        self._fitted_vols = None
        self._fitted_corr = None
        self._fitted_cov = None

        self._obs_cov = None

    def _fit(self, training_data: TrainingData) -> None:
        ret = training_data.simple_excess_returns

        self._fitted_cov = _QIS(ret, shrinkage=self.shrinkage, k=1)
        self._fitted_cov.index = self.available_assets
        self._fitted_cov.columns = self.available_assets

    def _predict(self, prediction_data: PredictionData) -> pd.DataFrame:
        return self._fitted_cov
