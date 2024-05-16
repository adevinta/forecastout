from forecastout.utils.update_config import update_config
import unittest

HORIZON = 10
MONTHS_TO_BACKTEST = 10
MODELS_TO_USE = ['test']


def test_random_forest_model():
    updated_config = update_config(
        input_dict,
        horizon=HORIZON,
        months_to_backtest=MONTHS_TO_BACKTEST,
        models_to_use=MODELS_TO_USE
    )
    unittest.TestCase().assertDictEqual(d1=updated_config, d2=expected_dict)


input_dict = {
    "horizon": 15,
    "months_to_backtest": 3,
    "models_to_use": ['autoarima', 'holtwinters', 'prophet'],
    "ensemble": {
      "ensemble_method": "average_top_models",
      "average_top_models": {"n_tops": 3}
    }
}

expected_dict = {
    "horizon": HORIZON,
    "months_to_backtest": MONTHS_TO_BACKTEST,
    "models_to_use": MODELS_TO_USE,
    "ensemble": {
      "ensemble_method": "average_top_models",
      "average_top_models": {"n_tops": 3}
    }
}
