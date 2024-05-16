import pandas as pd
from typing import List
from forecastout.data_engine.data_handler import DataHandler


class InputChecker:
    def __init__(
            self,
            df: pd.DataFrame,
            sum_aggregation: bool = True,
            horizon: int = None,
            months_to_backtest: int = None,
            models_to_use: List[str] = None,
            average_top_models_number: int = None
    ):
        # -- Main inputs
        self.df = df
        self.sum_aggregation = sum_aggregation
        self.horizon = horizon
        self.months_to_backtest = months_to_backtest
        self.models_to_use = models_to_use
        self.average_top_models_number = average_top_models_number
        # -- Preliminary Checks
        self.__check_df_instance()
        self.__check_df_column_names()
        self.__check_sum_aggregation()
        self.__check_horizon()
        self.__check_months_to_backtest()
        self.__check_models_to_use()
        self.__check_average_top_models_number()
        self.__check_backtesting_consistency()

    def __check_df_instance(self):
        if not isinstance(self.df, pd.DataFrame):
            error_string = (
                    "Input df should be a pandas DataFrame."
            )
            raise TypeError(error_string)

    def __check_df_column_names(self):
        column_names_list = ['date', 'value']
        for column in self.df.columns:
            if column not in column_names_list:
                error_string = (
                    f"Column name '{column}' for the input " +
                    "df is not acceptable. The acceptable " +
                    "column names for a DataFrame to be " +
                    f"accepted are: {column_names_list}."
                )
                raise ValueError(error_string)

    def __check_backtesting_consistency(self):
        # -- Checks of DF
        data_handler = DataHandler(
            df=self.df,
            horizon=self.horizon,
            sum_aggregation=self.sum_aggregation
        )
        df_monthly = data_handler.df_monthly.copy()
        # -- df_monthly
        num_of_monthly_observations = df_monthly.dropna()['date'].count()
        if num_of_monthly_observations - self.months_to_backtest < 24:
            error_string = (
                "Your data consists in  " +
                f"'{num_of_monthly_observations}' monthly observations, " +
                "while the input parameter months_to_backtest " +
                f"is set to '{self.months_to_backtest}'. " +
                "Make sure that the number of monthly observations " +
                "minus the input parameter months_to_backtest " +
                "is larger than 24."
            )

            raise ValueError(error_string)

    def __check_sum_aggregation(self):
        if not isinstance(self.sum_aggregation, bool):
            error_string = (
                "Input parameter sum_aggregation is " +
                f"{type(self.sum_aggregation)}. " +
                "Expected input type is bool. " +
                "If the df input presents daily data, you can aggregate it " +
                "monthly by adding it in a monthly basis if input " +
                "sum_aggregation is set to True " +
                "or average it if set to False."
            )
            raise TypeError(error_string)

    def __check_horizon(self):
        if not isinstance(self.horizon, int):
            error_string = (
                "Input parameter horizon is " +
                f"{type(self.horizon)}. " +
                "Expected input type is int."
            )
            raise TypeError(error_string)

    def __check_months_to_backtest(self):
        if not isinstance(self.months_to_backtest, int):
            error_string = (
                "Input parameter months_to_backtest is " +
                f"{type(self.months_to_backtest)}. " +
                "Expected input type is int."
            )
            raise TypeError(error_string)

    def __check_models_to_use(self):
        model_names_list = [
            'autoarima',
            'holtwinters',
            'prophet',
            'seasonalnaive'
        ]
        if not isinstance(self.models_to_use, list):
            error_string = (
                    "Input parameter models_to_use is " +
                    f"{type(self.models_to_use)}. " +
                    "Expected input type is List[str]."
            )
            raise TypeError(error_string)
        for model in self.models_to_use:
            if not isinstance(model, str):
                error_string = (
                    "Element of list models_to_use is a " +
                    f"{type(self.models_to_use)}. " +
                    "Expected elements of the list must be all str."
                )
                raise TypeError(error_string)
            if model not in model_names_list:
                error_string = (
                    f"Model name '{model}' is not acceptable. " +
                    f"The acceptable model names are: " +
                    f"{model_names_list}."
                )
                raise ValueError(error_string)

    def __check_average_top_models_number(self):
        if not isinstance(self.average_top_models_number, int):
            error_string = (
                "Input parameter average_top_models_number is "
                f"{type(self.average_top_models_number)}. " +
                "Expected input type is int."
            )
            raise TypeError(error_string)
