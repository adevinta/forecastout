from typing import List


def update_config(
        config: dict,
        horizon: int = None,
        months_to_backtest: int = None,
        models_to_use: List[str] = None,
        average_top_models_number: int = None
) -> dict:
    if horizon is not None:
        config["horizon"] = horizon
    if months_to_backtest is not None:
        config["months_to_backtest"] = months_to_backtest
    if models_to_use is not None:
        config["models_to_use"] = models_to_use
    if average_top_models_number is not None:
        config["ensemble"]["average_top_models"]["n_tops"] = \
            average_top_models_number
    if (average_top_models_number is not None) and\
            (average_top_models_number > len(models_to_use)):
        config["ensemble"]["average_top_models"]["n_tops"] = \
            len(models_to_use)
    return config
