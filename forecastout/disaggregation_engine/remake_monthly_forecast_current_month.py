import pandas as pd


def remake_monthly_forecast_current_month(
        df_daily_forecast: pd.DataFrame,
        df_actuals: pd.DataFrame
) -> pd.DataFrame:
    # -- Get actuals of unfinished month
    first_forecast_day = df_daily_forecast['date'].min()
    first_actuals_day = pd.to_datetime(
        str(first_forecast_day)[0:7] + '-01'
    )
    df_actuals = df_actuals.loc[
        (df_actuals['date'] >= first_actuals_day) &
        (df_actuals['date'] < first_forecast_day)
        ].copy()
    df_actuals['forecast'] = df_actuals['value'].copy()
    df_actuals['forecast_lower'] = df_actuals['value'].copy()
    df_actuals['forecast_upper'] = df_actuals['value'].copy()
    df_actuals['model'] = df_daily_forecast['model'].unique()[0]
    df_actuals.drop('value', axis=1, inplace=True)
    # -- Recompute monthly forecast
    df_daily_forecast = pd.concat(
        [df_actuals, df_daily_forecast], axis=0
    )
    df_daily_forecast["date"] = (
        df_daily_forecast["date"].astype(str).str[0:7]
    )
    df_monthly_forecast = (
        df_daily_forecast
        .groupby(["date", "model"])
        [["forecast", "forecast_lower", "forecast_upper"]]
        .sum()
        .reset_index()
    )
    df_monthly_forecast['date'] = pd.to_datetime(
        df_monthly_forecast['date'] + '-01'
    )
    column_order = [
        "forecast",
        "forecast_lower",
        "forecast_upper",
        "date",
        "model"
    ]
    df_monthly_forecast = df_monthly_forecast[column_order].copy()
    return df_monthly_forecast
