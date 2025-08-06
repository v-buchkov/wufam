from dataclasses import dataclass


@dataclass
class Dataset:
    BM_6_D = "6_Portfolios_2x3_Daily.csv"
    BM_25_D = "25_Portfolios_5x5_Daily.csv"
    BM_100_D = "100_Portfolios_10x10_Daily.csv"

    BM_6_M = "6_Portfolios_2x3.csv"
    BM_25_M = "25_Portfolios_5x5.csv"
    BM_100_M = "100_Portfolios_10x10.csv"

    FACTORS_D = "F-F_Research_Data_Factors_daily.csv"
    FACTORS_M = "F-F_Research_Data_Factors.csv"
