import pandas as pd
from forecastout.disaggregation_features.feature_creator \
    import FeatureCreator


class DisaggregationModelPredictor:
    def __init__(self, model, df: pd.DataFrame, config: dict):
        self.model = model
        self.df = df
        self.config = config

    def predict(self) -> pd.DataFrame:
        list_output = []
        df_copy = self.df.copy()
        list_dates_test = df_copy.loc[df_copy['test'] == 1, "date"].tolist()
        for day in list_dates_test:
            # -- Get test
            df_test_x_day = (
                df_copy.loc[
                    df_copy['date'] == day,
                    self.config['disaggregation_features']
                ].copy()
            )
            # -- Do forecast
            df_prediction_i = self.model.do_prediction(
                df_test_x=df_test_x_day,
                list_dates=[day])
            # -- append results
            list_output.append(df_prediction_i)
            # -- UPDATE FEATURES
            df_copy.loc[
                df_copy['date'] == day, "value"
            ] = df_prediction_i['forecast'].values[0]
            feature_creator = FeatureCreator(df=df_copy)
            feature_creator.get_lags(
                features=["value"],
                number_lags=7
            )
            feature_creator.get_moving_average(
                features=['value'],
                windows=[7, 30]
            )
            df_copy = feature_creator.df.copy()
        # -- Convert to df
        df_output = pd.concat(list_output)
        # -- Ensure positivity
        df_num = df_output._get_numeric_data()
        df_num[df_num < 0] = 0
        return df_output
