import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta


class TrainTestSplit:
    """
    This class identifies which observations are for training and for testing
    """

    @staticmethod
    def split_by_current_closed_month(
            df: pd.DataFrame,
            current_month: str,
            granularity_month: bool = True
    ) -> pd.DataFrame:
        min_date = (
            pd.to_datetime(current_month)
            .replace(day=1, hour=0, minute=0, second=0)
        )
        if granularity_month:
            df['test'] = np.where(df["date"] <= min_date, 0, 1)
        else:
            df['test'] = np.where(df["date"] < min_date, 0, 1)
        return df

    @staticmethod
    def split_for_ts_backtest(
            df: pd.DataFrame,
            current_month: str,
            months_to_backtest: int
    ) -> pd.DataFrame:
        # -- Filter data before current month
        current_month = (
            pd.to_datetime(current_month)
            .replace(day=1, hour=0, minute=0, second=0)
        )
        df = df[df["date"] <= current_month].copy()
        # -- Get train/test data
        min_test_date = (
                current_month - relativedelta(months=months_to_backtest)
        )
        df['test'] = np.where(df["date"] <= min_test_date, 0, 1)
        # -- Return
        return df
