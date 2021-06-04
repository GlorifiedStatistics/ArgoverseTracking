from ForecastModels.ForecastModel import *
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, MultiTaskElasticNet, TweedieRegressor, SGDRegressor, \
    PassiveAggressiveRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor


class SKLearnModel(ForecastModel):
    def __init__(self, n_jobs=20, max_data=10_000, norm_func='noop', **kwargs):
        self.n_jobs = n_jobs
        super().__init__(norm_func=norm_func, max_data=max_data, **kwargs)

    def save(self):
        with open(self._model_path(), 'wb') as f:
            pickle.dump(self.model, f)

    def load(self):
        with open(self._model_path(), 'rb') as f:
            return pickle.load(f)


class RandomForestModel(SKLearnModel):
    def __init__(self, n_estimators=100, max_depth=20, **kwargs):
        self.n_estimators, self.max_depth = n_estimators, max_depth
        super().__init__(max_depth=max_depth, **kwargs)
        self.name += "_".join([str(n_estimators), str(max_depth)])

    def _gen_model(self):
        return RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth, n_jobs=self.n_jobs)


class DecisionTreeModel(SKLearnModel):
    def __init__(self, max_depth=20, **kwargs):
        self.max_depth = max_depth
        super().__init__(**kwargs)
        self.name += "_" + str(max_depth)

    def _gen_model(self):
        return DecisionTreeRegressor(max_depth=self.max_depth)


class LinearRegressionModel(SKLearnModel):
    def _gen_model(self):
        return LinearRegression(n_jobs=self.n_jobs)


class ElasticNetModel(SKLearnModel):
    def _gen_model(self):
        return MultiTaskElasticNet()


class GeneralLinearModel(SKLearnModel):
    def __init__(self, power=0, max_iter=1_000, **kwargs):
        self.power, self.max_iter = power, max_iter
        super().__init__(**kwargs)
        self.name += "_%f_%d" % (power, max_iter)

    def _gen_model(self):
        return MultiOutputRegressor(TweedieRegressor(power=self.power, max_iter=self.max_iter), n_jobs=self.n_jobs)


class SGDRegressorModel(SKLearnModel):
    def __init__(self, loss='squared_loss', penalty='l2', **kwargs):
        self.loss, self.penalty = loss, penalty
        super().__init__(**kwargs)
        self.name += "_" + loss

    def _gen_model(self):
        return MultiOutputRegressor(SGDRegressor(loss=self.loss, penalty=self.penalty, max_iter=10_000),
                                    n_jobs=self.n_jobs)


class PassiveAggressiveModel(SKLearnModel):
    def _gen_model(self):
        return MultiOutputRegressor(PassiveAggressiveRegressor(), n_jobs=self.n_jobs)


class AdaBoostModel(SKLearnModel):
    def __init__(self, max_depth=20, **kwargs):
        self.base = DecisionTreeRegressor(max_depth=max_depth)
        super().__init__(**kwargs)
        self.name += "_" + str(max_depth)

    def _gen_model(self):
        return MultiOutputRegressor(AdaBoostRegressor(base_estimator=self.base), n_jobs=self.n_jobs)


class XGBRegressorModel(SKLearnModel):
    def _gen_model(self):
        return MultiOutputRegressor(XGBRegressor(), n_jobs=1)


class GaussianProcessModel(SKLearnModel):
    def _gen_model(self):
        return GaussianProcessRegressor()


class KernelRidgeModel(SKLearnModel):
    def __init__(self, kernel='linear', **kwargs):
        self.kernel = kernel
        super().__init__(**kwargs)
        self.name += "_" + kernel

    def _gen_model(self):
        return KernelRidge(kernel=self.kernel)


class KNearestNeighborsModel(SKLearnModel):
    def __init__(self, k=3, weights='uniform', p=2, **kwargs):
        self.k, self.weights, self.p = k, weights, p
        super().__init__(**kwargs)
        self.name += "_".join(["", str(k), weights, str(p)])

    def _gen_model(self):
        return KNeighborsRegressor(self.k, weights=self.weights, p=self.p, n_jobs=self.n_jobs)




