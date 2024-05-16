import pandas as pd
from statsmodels.tsa.seasonal import STL


def create_df_ts_decomposition(
        df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create DF with stl decomposition
    """
    df_hist = df[df['test'] == 0].copy()
    df_hist = pd.Series(
        df_hist['value'].to_list(),
        index=pd.date_range(
            start=df_hist['date'].min(),
            end=df_hist['date'].max(),
            freq="MS"),
        name="ts"
    )
    stl = STL(df_hist, period=12)
    res = stl.fit()
    df_ts_decomposition = pd.DataFrame(
        dict(
            s1=res.seasonal,
            s2=res.trend,
            s3=res.resid
        )
    ).reset_index()
    df_ts_decomposition.columns = ['date', 'seasonal', 'trend', 'residual']
    return df_ts_decomposition
