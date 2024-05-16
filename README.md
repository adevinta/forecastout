# forecastout

ForecastOut makes predictions for univariate time series using both statistical 
and machine learning approaches.

The algorithm works for both monthly and daily data.


In the case of monthly data, ForecastOut makes predictions using different 
statistical models (ARIMA, variations of Holt-Winters, Prophet, etc.) and then 
ensembles them using backtesting. A Naive Seasonal model is averaged into the 
resulting prediction in case the latter is not plausible.

For daily data, ForecastOut first aggregates the data (by summing or averaging) 
on a monthly basis and applies the algorithm described above to make a monthly 
forecast. Then, it disaggregates the monthly forecast to a daily basis using 
the time series and calendar information in a Random Forest.

The algorithm provides not only the forecast output and 95% confidence bands 
but also detailed results of each model, backtesting, and time series 
decomposition.


## Installation

```bash

pip install forecastout

```

## Quickstart

To make a prediction, you only need one necessary input: a pandas DataFrame 
with two columns:

- date: string format "YYYY-MM-DD" (e.g., "2024-01-01").
- value: integer or float format (e.g., 1000.0).

It should contain at least 24 monthly observations (if the data is daily,
it should cover more than 24 months).

```{python, error=TRUE, include=TRUE}
import pandas as pd
from forecastout import ForecastOut

forecastout = ForecastOut(
    df=pd.DataFrame()
)
```

This will directly execute a prediction for 15 months for the inserted time 
series input. You can access the results as follows:

```{python, error=TRUE, include=TRUE}
# -- Monthly forecast output
print(forecastout.df_monthly_forecast)
# -- Monthly forecast predictions by model
print(forecastout.df_predictions_by_model) 
# -- Daily forecast output (if daily data)
print(forecastout.df_daily_forecast)
 # -- Time series decomposition
print(forecastout.df_ts_decomposition)
 # -- Backtesting predictions for each model
print(forecastout.df_predictions_bt)
 # -- Daily decomposition percentage (if daily data)
print(forecastout.df_daily_shares)
```

## Inputs
ForecastOut can customize the results through the following parameters. 

```{python, error=TRUE, include=TRUE}
import pandas as pd
from forecastout import ForecastOut

    forecastout = ForecastOut(
        df=pd.DataFrame(),
        sum_aggregation=True,
        horizon=3,
        months_to_backtest=3,
        models_to_use=['autoarima', 'holtwinters', 'prophet'],
        average_top_models_number=1
    )
```

- sum_aggregation (bool): If True, it adds daily data. If False, it averages daily data.
- horizon (int): Number of periods to forecast.
- months_to_backtest (int): Number of periods to do the backtesting.
- models_to_use (list): List of models to be used. ForecastOut has available 'autoarima', 'holtwinters' and 'prophet' (new coming soon).
- average_top_models_number (int): Number of models to use as average (if 1, it selects the most precise model in the backtesting).

## Remarks
- Calendar Data available is only for Spain when doing the disaggregation from monthly to daily data (more calendars will be available in the future).
- In case the input DataFrame contains gaps, they will be assumed to be 0 value.
