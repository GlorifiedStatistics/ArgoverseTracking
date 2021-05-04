from ForecastModels.ForecastModel import *
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, MultiTaskElasticNet, TweedieRegressor, SGDRegressor, \
    PassiveAggressiveRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor


class SKLearnModel(ForecastModel):
    def __init__(self, n_jobs=20, max_data=10_000, norm_func='noop', **kwargs):
        super().__init__(norm_func, n_jobs=n_jobs, max_data=max_data, **kwargs)

    def _gen_model(self, **kwargs):
        pass

    def _train_model(self):
        self.model.fit(self.data['train_x'], self.data['train_y'])

    def save(self):
        with open(self._model_path(), 'wb') as f:
            pickle.dump(self.model, f)

    def load(self):
        with open(self._model_path(), 'rb') as f:
            return pickle.load(f)


class RandomForestModel(SKLearnModel):
    def __init__(self, n_estimators=100, max_depth=20, **kwargs):
        super().__init__(n_estimators=n_estimators, max_depth=max_depth, **kwargs)
        self.name = "_".join([str(n_estimators), str(max_depth), self.name])

    def _gen_model(self, **kwargs):
        return RandomForestRegressor(n_estimators=kwargs['n_estimators'], max_depth=kwargs['max_depth'],
                                     n_jobs=kwargs['n_jobs'])


class DecisionTreeModel(SKLearnModel):
    def __init__(self, max_depth=20, **kwargs):
        super().__init__(max_depth=max_depth, **kwargs)
        self.name = "_".join([str(max_depth), self.name])

    def _gen_model(self, **kwargs):
        return DecisionTreeRegressor(max_depth=kwargs['max_depth'])


class LinearRegressionModel(SKLearnModel):
    def _gen_model(self, **kwargs):
        return LinearRegression(n_jobs=kwargs['n_jobs'])


class ElasticNetModel(SKLearnModel):
    def _gen_model(self, **kwargs):
        return MultiTaskElasticNet()


class GeneralLinearModel(SKLearnModel):
    def __init__(self, power=0, max_iter=1_000, **kwargs):
        super().__init__(power=power, max_iter=max_iter, **kwargs)
        self.name = "%f_%d" % (power, max_iter) + self.name

    def _gen_model(self, **kwargs):
        return MultiOutputRegressor(TweedieRegressor(power=kwargs['power'], max_iter=kwargs['max_iter']),
                                    n_jobs=kwargs['n_jobs'])


class SGDRegressorModel(SKLearnModel):
    def __init__(self, loss='squared_loss', penalty='l2', **kwargs):
        super().__init__(loss=loss, penalty=penalty, **kwargs)
        self.name = loss + "_" + self.name

    def _gen_model(self, **kwargs):
        return MultiOutputRegressor(SGDRegressor(loss=kwargs['loss'], penalty=kwargs['penalty'], max_iter=10_000),
                                    n_jobs=kwargs['n_jobs'])


class PassiveAggressiveModel(SKLearnModel):
    def _gen_model(self, **kwargs):
        return MultiOutputRegressor(PassiveAggressiveRegressor(), n_jobs=kwargs['n_jobs'])


class AdaBoostModel(SKLearnModel):
    def __init__(self, max_depth=20, **kwargs):
        super().__init__(base=DecisionTreeRegressor(max_depth=max_depth), **kwargs)
        self.name = str(max_depth) + "_" + self.name

    def _gen_model(self, **kwargs):
        return MultiOutputRegressor(AdaBoostRegressor(base_estimator=kwargs['base']), n_jobs=kwargs['n_jobs'])


class GaussianProcessModel(SKLearnModel):
    def _gen_model(self, **kwargs):
        return GaussianProcessRegressor()


class KernelRidgeModel(SKLearnModel):
    def __init__(self, kernel='linear', **kwargs):
        super().__init__(kernel=kernel, **kwargs)
        self.name = kernel + "_" + self.name

    def _gen_model(self, **kwargs):
        return KernelRidge()


class KNearestNeighborsModel(SKLearnModel):
    def __init__(self, k=3, weights='uniform', p=2, **kwargs):
        super().__init__(k=k, weights=weights, p=p, **kwargs)
        self.name = str(k) + "_" + weights + "_" + str(p) + "_" + self.name

    def _gen_model(self, **kwargs):
        return KNeighborsRegressor(kwargs['k'], weights=kwargs['weights'], p=kwargs['p'], n_jobs=kwargs['n_jobs'])




