import pandas as pd


def get_daily_shares(
        df_daily_prediction: pd.DataFrame,
        sum_aggregation: bool
) -> pd.DataFrame:
    df_daily_prediction['year_month'] = (
        df_daily_prediction["date"].astype(str).str[0:7]
    )
    df_daily_prediction['monthly_totals'] = (
        df_daily_prediction
        .groupby(["year_month"])["forecast"].transform(sum)
    )
    if sum_aggregation:
        df_daily_prediction["daily_share"] = (
                df_daily_prediction["forecast"] /
                df_daily_prediction["monthly_totals"]
        )
    else:
        df_daily_prediction["days_month"] = (
            df_daily_prediction
            .groupby(["year_month"])["date"].transform('count')
        )
        df_daily_prediction["daily_share"] = (
                df_daily_prediction["days_month"] *
                df_daily_prediction["forecast"] /
                df_daily_prediction['monthly_totals']
        )
    df_daily_shares = df_daily_prediction[["date", "daily_share"]]
    return df_daily_shares
