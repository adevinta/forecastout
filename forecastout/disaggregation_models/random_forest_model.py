from forecastout.disaggregation_models.abstract_model \
    import DisaggregationModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RepeatedKFold
import pandas as pd


class RandomForestModel(DisaggregationModel):
    """
    Random Forest Model
    """
    def __init__(self,
                 *args,
                 **kwargs
                 ):
        super(RandomForestModel, self).__init__(*args, **kwargs)
        # -- grid search
        grid = GridSearchCV(
            estimator=RandomForestRegressor(random_state=123),
            param_grid=self.dict_config['param_grid'],
            scoring=self.dict_config['scoring'],
            n_jobs=1,
            cv=RepeatedKFold(
                n_splits=self.dict_config['n_splits'],
                n_repeats=self.dict_config['n_repeats'],
                random_state=123
            ),
            refit=self.dict_config['refit'],
            verbose=0,
            return_train_score=self.dict_config['return_train_score']
        )
        grid.fit(X=self.df_train_x, y=self.df_train_y.values.ravel())
        self.random_forest_model = grid.best_estimator_

    def do_prediction(
            self, 
            list_dates: list, 
            df_test_x: pd.DataFrame) -> pd.DataFrame:
        list_prediction = self.random_forest_model.predict(df_test_x).tolist()
        df_prediction = pd.concat(
            [
                pd.DataFrame(list_dates, columns=['date']),
                pd.DataFrame(list_prediction, columns=['forecast']),
            ],
            axis=1,
        )
        df_prediction['model'] = 'random_forest'
        df_prediction = df_prediction[['forecast', 'model', 'date']].copy()
        return df_prediction

    def get_feature_importance(self) -> pd.DataFrame:
        list_feature_importance = self.random_forest_model.feature_importances_
        df_feature_importance = pd.concat(
            [pd.DataFrame(self.df_train_x.columns,
                          columns=['feature']),
             pd.DataFrame(list_feature_importance,
                          columns=['feature_importance'])
             ], axis=1
        )
        df_feature_importance['model'] = 'random_forest'
        df_feature_importance.sort_values(
            'feature_importance',
            ascending=False,
            inplace=True
        )
        return df_feature_importance
