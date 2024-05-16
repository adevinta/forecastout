import pandas as pd
from forecastout.disaggregation_features.feature_creator \
    import FeatureCreator
from forecastout.disaggregation_features.feature_normalizer \
    import FeatureNormalizer
from forecastout.disaggregation_features.feature_encoding \
    import FeatureEncoder


def add_features(
        df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create list of features
    """
    # -- Create features
    feature_creator = FeatureCreator(df=df)
    feature_creator.get_day()
    feature_creator.get_month()
    feature_creator.get_year()
    feature_creator.get_day_of_week()
    feature_creator.get_holidays()
    # -- Normalize value
    df = FeatureNormalizer.normalization_base100(
        df=feature_creator.df,
        norm_level=["month", "year"],
        norm_feats=["value"]
    )
    # -- Create more features
    feature_creator = FeatureCreator(df=df)
    feature_creator.get_lags(
        features=["value"],
        number_lags=7
    )
    feature_creator.get_moving_average(
        features=['value'],
        windows=[7, 30]
    )
    df = feature_creator.df.copy()
    # -- Encode features
    feature_encoder = FeatureEncoder()
    df = feature_encoder.numerical_encode(
        df=df.copy(),
        encode_feats=['day', 'day_of_week', 'month'],
        encode_value='value'
    )
    return df
