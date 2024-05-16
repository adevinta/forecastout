from forecastout.disaggregation_models.random_forest_model \
    import RandomForestModel
import pandas as pd


def test_random_forest_model():
    model = RandomForestModel(
        df_train_y=input_train_df()['value'],
        df_train_x=input_train_df()[['feature_1', 'feature_2', 'feature_3']],
        dict_config=config_input_dict())
    df_prediction = model.do_prediction(
        list_dates=input_test_df()['date'].tolist(),
        df_test_x=input_test_df()[['feature_1', 'feature_2', 'feature_3']])
    df_feature_importance = model.get_feature_importance()
    pd.testing.assert_frame_equal(
        df_prediction,
        expected_prediction_df()
    )
    pd.testing.assert_frame_equal(
        df_feature_importance.reset_index(drop=True),
        expected_feature_importance_df().reset_index(drop=True)
    )


def config_input_dict():
    config = {
        "param_grid":
            {'n_estimators': [150],
             'max_features': [2],
             'max_depth': [3]
             },
        "scoring": 'neg_root_mean_squared_error',
        "n_splits": 5,
        "n_repeats": 3,
        "refit": True,
        "return_train_score": True
    }
    return config


def input_train_df():
    date_list = [
        "2021-01-01",
        "2021-02-01",
        "2021-03-01",
        "2021-04-01",
        "2021-05-01",
        "2021-06-01",
        "2021-07-01",
        "2021-08-01",
        "2021-09-01",
        "2021-10-01",
        "2021-11-01",
        "2021-12-01",
        "2022-01-01",
        "2022-02-01",
        "2022-03-01",
        "2022-04-01",
        "2022-05-01",
        "2022-06-01",
        "2022-07-01",
        "2022-08-01",
        "2022-09-01",
        "2022-10-01",
        "2022-11-01",
        "2022-12-01",
        "2023-01-01",
        "2023-02-01",
        "2023-03-01",
        "2023-04-01",
        "2023-05-01",
        "2023-06-01",
        "2023-07-01",
        "2023-08-01",
        "2023-09-01",
        "2023-10-01",
        "2023-11-01",
        "2023-12-01"
    ]
    value_list1 = [
        5830302.908,
        7400457.853,
        7449702.162,
        6480787.829,
        6995587.495,
        1675807.097,
        8372644.865,
        11324372.84,
        1799333.476,
        12702195.21,
        4509435.271,
        13316086.92,
        12826666.4,
        15473684.6,
        14899404.32,
        12463053.52,
        12991805.35,
        3016452.775,
        14652128.51,
        19318047.79,
        2998889.127,
        20724634.3,
        7215096.433,
        20925279.44,
        19823029.89,
        23546911.35,
        22349106.49,
        18445319.21,
        18988023.2,
        4357098.453,
        20931612.16,
        27311722.74,
        4198444.778,
        28747073.38,
        9920757.596,
        28534471.96,
    ]
    return pd.DataFrame(
        {'date': pd.to_datetime(date_list),
         'value': value_list1,
         'feature_1': [value * 1.1 for value in value_list1],
         'feature_2': [value / 2 for value in value_list1],
         'feature_3': [value ** 3 for value in value_list1]}
    )


def input_test_df():
    date_list = [
        "2024-01-01",
        "2024-02-01",
        "2024-03-01"
    ]
    value_list = [
        5830302.908,
        7400457.853,
        7449702.162
    ]
    return pd.DataFrame(
        {'date': pd.to_datetime(date_list),
         'value': value_list,
         'feature_1': [value * 1.08 for value in value_list],
         'feature_2': [value / 2.1 for value in value_list],
         'feature_3': [value ** 3.1 for value in value_list]}
    )


def expected_prediction_df():
    value_list = [
        7417858.57572,
        8704999.19365,
        8704999.19365

    ]
    model_list = [
        "random_forest",
        "random_forest",
        "random_forest"
    ]
    date_list = [
        "2024-01-01",
        "2024-02-01",
        "2024-03-01"
    ]
    return pd.DataFrame({
        'forecast': value_list,
        'model': model_list,
        'date': pd.to_datetime(date_list)
    })


def expected_feature_importance_df():
    feature_list = [
        "feature_2",
        "feature_1",
        "feature_3"
    ]
    feature_importance_list = [
        0.345383,
        0.327562,
        0.327055
    ]
    model_list = [
        "random_forest",
        "random_forest",
        "random_forest"
    ]
    return pd.DataFrame({
        'feature': feature_list,
        'feature_importance': feature_importance_list,
        'model': model_list
    })
