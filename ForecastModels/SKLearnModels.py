from ForecastModels.ForecastModel import *


class SKLearnModel(ForecastModel):
    def __init__(self, redo=False, norm_func=None, y_output='full', n_estimators=100, max_depth=20, n_jobs=-1,
                 max_data=100_000, normalize=False, criterion='mse',
                 random_state=RANDOM_STATE, **kwargs):

        super().__init__(redo, norm_func, y_output, n_estimators=n_estimators, max_depth=max_depth, n_jobs=n_jobs,
                         normalize=normalize, random_state=random_state, criterion=criterion,
                         **kwargs)

        # Take into account the max_data in case things get too large
        ms = min(len(self.data['train_x']), max_data)
        self.data['train_x'] = self.data['train_x'][:ms]
        self.data['train_y'] = self.data['train_y'][:ms]

    def _gen_model(self, **kwargs):
        pass

    def _train_model(self):
        self.model.fit(self.data['train_x'], self.data['train_y'])


class RandomForestModel(SKLearnModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _gen_model(self, **kwargs):
        return RandomForestRegressor(n_estimators=kwargs['n_estimators'], max_depth=kwargs['max_depth'],
                                     n_jobs=kwargs['n_jobs'], random_state=kwargs['random_state'])


class LinearRegressionModel(SKLearnModel):
    def __init__(self, max_data=10_000, **kwargs):
        super().__init__(max_data=max_data, **kwargs)

    def _gen_model(self, **kwargs):
        return LinearRegression(normalize=kwargs['normalize'], n_jobs=kwargs['n_jobs'])


class DecisionTreeModel(SKLearnModel):
    def __init__(self, max_data=10_000, **kwargs):
        super().__init__(max_data=max_data, **kwargs)

    def _gen_model(self, **kwargs):
        return DecisionTreeRegressor(max_depth=kwargs['max_depth'], random_state=kwargs['random_state'],
                                     criterion=kwargs['criterion'])



