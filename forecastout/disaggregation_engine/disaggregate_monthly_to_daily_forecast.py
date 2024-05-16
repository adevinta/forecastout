import pandas as pd


def disaggregate_monthly_to_daily_forecast(
        df_daily_shares: pd.DataFrame,
        df_monthly_forecast: pd.DataFrame
) -> pd.DataFrame:
    df_daily_shares["year_month"] = df_daily_shares["date"].astype(
        str).str[0:7]
    df_monthly_forecast["year_month"] = df_monthly_forecast[
                                            "date"].astype(str).str[
                                        0:7]
    df_monthly_forecast.drop('date', axis=1, inplace=True)
    df_daily_forecast = df_daily_shares.merge(
        df_monthly_forecast,
        on='year_month',
        how='left'
    )
    forecast_columns = ["forecast", "forecast_lower", "forecast_upper"]
    for forecast_column in forecast_columns:
        df_daily_forecast[forecast_column] = df_daily_forecast[
                                                 forecast_column] * \
                                             df_daily_forecast[
                                                 "daily_share"]
    relevant_columns = ["date"] + forecast_columns + ["model"]
    df_daily_forecast = df_daily_forecast[relevant_columns].copy()
    return df_daily_forecast
