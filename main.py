from Visualize import *
from ForecastModels.Models import *

from Argoverse.NormFuncs import get_norm_funcs

def do_model():

    #model = RandomForestModel(max_depth=None, n_estimators=100, max_data=1_000, data_type='step', norm_func='standardize', include_lanes=False, extra_features=False)
    #model = MLPModel(norm_func='tanh', batch_norm=True, act='tanh', epochs=10, dp=None, hidden_layers=3, batch_size=512, y_norm_func='tanh')
    #model = MLPModel(norm_func='linear', batch_norm=True, act='sigmoid', epochs=7, dp=None, hidden_layers=3, batch_size=512, loss='mse')
    #model = LinearRegressionModel(norm_func='noop', data_type='raw')
    #model = GeneralLinearModel(**{'max_data': 50_000, 'data_type': 'step', 'norm_func': 'standardize', 'power': 0})
    model = SGDRegressorModel(**{'max_data': None, 'data_type': 'raw', 'norm_func': 'standardize', 'loss': 'huber', 'penalty': 'l1', 'load_model':True})
    #model = PassiveAggressiveModel(norm_func='std_step', max_data=10_000, data_type='step')
    #model = KNearestNeighborsModel(k=10, p=1, data_type='step', norm_func='std_step', max_data=10_000)
    #model = ElasticNetModel(**{'max_data': None, 'data_type': 'step', 'norm_func': 'linear'})

    #model = AdaBoostModel(max_data=50_000, max_depth=25, data_type='step', norm_func='linear', include_lanes=False)

    #model.train()
    #model.compute_val()

    #model.test_predictions()

do_model()

from TestModels import test_models

#test_models('best_nn')
