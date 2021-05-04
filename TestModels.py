from utils.Utils import *
from ForecastModels.Models import *
from itertools import product
from datetime import datetime

_M = 1000000000000000000000000


_DATA_TYPES = ['step', 'raw']
_NORM_FUNCS = ['noop', 'linear', 'std_step', 'standardize']
_INCLUDE_LANES = [False, True]
_EXTRA_FEATS = [False, True]
_MD = 10_000

# 2-tuple of model: the class of model
#   kwargs: dictionary of kwarg name, and value/list of values (if list, then all will be done in grid based search)
INITIAL_SKLEARN_MODELS = (
    (RandomForestModel, {
        'n_estimators': 100, 'max_depth': None, 'max_data': _MD, 'data_type': _DATA_TYPES, 'norm_func': _NORM_FUNCS
    }),
    (DecisionTreeModel, {
        'max_depth': None, 'max_data': _MD, 'data_type': _DATA_TYPES, 'norm_func': _NORM_FUNCS
    }),
    (LinearRegressionModel, {
        'max_data': _MD, 'data_type': _DATA_TYPES, 'norm_func': _NORM_FUNCS
    }),
    (ElasticNetModel, {
        'max_data': _MD, 'data_type': _DATA_TYPES, 'norm_func': _NORM_FUNCS
    }),
    (GeneralLinearModel, {
        'max_data': _MD, 'data_type': _DATA_TYPES, 'norm_func': _NORM_FUNCS, 'power': [0, 1, 1.5, 2, 3]
    }),
    (SGDRegressorModel, {
        'max_data': _MD, 'data_type': _DATA_TYPES, 'norm_func': _NORM_FUNCS, 'loss': ['squared_loss', 'huber'],
        'penalty': ['l1', 'l2']
    }),
    (PassiveAggressiveModel, {
        'max_data': _MD, 'data_type': _DATA_TYPES, 'norm_func': _NORM_FUNCS
    }),
    (GaussianProcessModel, {
        'max_data': _MD, 'data_type': _DATA_TYPES, 'norm_func': _NORM_FUNCS
    }),
    (KernelRidgeModel, {
        'max_data': _MD, 'data_type': _DATA_TYPES, 'norm_func': _NORM_FUNCS
    }),
    (KNearestNeighborsModel, {
        'max_data': _MD, 'data_type': _DATA_TYPES, 'norm_func': _NORM_FUNCS, 'k': [1, 3, 5, 10], 'p': [1, 2]
    }),
)


_BDT = 'step'
_BNF = 'linear'
SECONDARY_SKLEARN_MODELS = (
    (RandomForestModel, {
        'n_estimators': 100, 'max_depth': None, 'max_data': _MD, 'data_type': _BDT, 'norm_func': _BNF,
        'include_lanes': _INCLUDE_LANES, 'extra_features': _EXTRA_FEATS
    }),
    (RandomForestModel, {
        'n_estimators': [10, 50, 100, 300, 500], 'max_depth': [2, 5, 10, 25, None], 'max_data': _MD,
        'data_type': _BDT, 'norm_func': _BNF
    }),
    (AdaBoostModel, {
        'max_depth': [5, 25, None], 'max_data': _MD, 'data_type': _BDT, 'norm_func': _BNF
    }),
)


VMIN = -1
VMAX = 1
STD_MIN = -0.5
STD_MAX = 0.5
INITIAL_NN_MODELS = (
    (MLPModel, {
        'epochs': 20, 'data_type': 'raw', 'norm_func': ['linear'], 'batch_norm': [True],
        'dp': [None], 'act': ['elu'], 'kernel_regularizer': [None],
        'include_lanes': True, 'extra_features': True, 'vmin': VMIN, 'vmax': VMAX, 'hidden_layers': [3]
    }),
    (MLPModel, {
        'epochs': 20, 'data_type': 'step', 'norm_func': ['std_step', 'linear'], 'batch_norm': [True],
        'dp': [None], 'act': ['elu', 'tanh'], 'kernel_regularizer': [None],
        'include_lanes': True, 'extra_features': True, 'vmin': VMIN, 'vmax': VMAX, 'std_min': STD_MIN, 'std_max': STD_MAX,
        'hidden_layers': [3]
    }),
    (MLPModel, {
        'epochs': 20, 'data_type': 'step', 'norm_func': ['tanh'], 'batch_norm': [True],
        'dp': [None], 'act': ['tanh'], 'kernel_regularizer': [None],
        'include_lanes': True, 'extra_features': True, 'vmin': VMIN, 'vmax': VMAX, 'std_min': STD_MIN, 'std_max': STD_MAX,
        'hidden_layers': [3]
    }),
    (MLPModel, {
        'epochs': 20, 'data_type': 'step', 'norm_func': ['linear', 'tanh', 'std_step'], 'batch_norm': [True],
        'dp': [None], 'act': ['sigmoid'], 'kernel_regularizer': [None],
        'include_lanes': True, 'extra_features': True, 'vmin': 0, 'vmax': 1, 'hidden_layers': [3], 'std_min': 0.25,
        'std_max': 0.75
    }),
)

SECONDARY_NN_MODELS = (
    (MLPModel, {
        'epochs': 20, 'data_type': 'step', 'norm_func': 'tanh', 'batch_norm': [True],
        'dp': [None, 0.2, 0.4, 0.8], 'act': ['tanh'], 'kernel_regularizer': [None, 'l1', 'l2', 'l1_l2'],
        'include_lanes': True, 'extra_features': True, 'hidden_layers': [2, 3, 5], 'vmin': VMIN, 'vmax': VMAX
    }),
)

FINAL_NN_MODELS = (
    (MLPModel, {
        'epochs': 20, 'data_type': 'step', 'norm_func': 'tanh', 'batch_norm': [True],
        'dp': [0.2], 'act': ['tanh'], 'include_lanes': True, 'extra_features': True, 'hidden_layers': [6, 7, 10, 15]
    }),
    (MLPModel, {
        'epochs': 20, 'data_type': 'step', 'norm_func': 'tanh', 'batch_norm': [True],
        'dp': [0.2], 'act': ['tanh'], 'include_lanes': True, 'extra_features': True,
        'layer_sizes': [  # Incoming size is: 6100
            [8000, 4000, 2000, 500, 200],
            [12000, 6000, 3000, 1000, 500, 200]
        ]
    }),
    (RNNModel, {
        'epochs': 20, 'data_type': 'step', 'norm_func': 'tanh', 'batch_norm': [True],
        'rnn_type': ['rnn', 'lstm', 'gru'], 'dp': [0.2], 'act': ['tanh'], 'include_lanes': [False],
        'extra_features': [False], 'cell_neurons': [256]
    }),
    (RNNModel, {
        'epochs': 20, 'data_type': 'step', 'norm_func': 'tanh', 'batch_norm': [True],
        'rnn_type': ['lstm'], 'dp': [0.2], 'act': ['tanh'], 'include_lanes': [True],
        'extra_features': [True], 'cell_neurons': [256], 'max_data': 50_000,
    }),
    (RNNModel, {
        'epochs': 20, 'data_type': 'step', 'norm_func': 'tanh', 'batch_norm': [True],
        'rnn_type': ['lstm'], 'dp': [0.2], 'act': ['tanh'], 'include_lanes': [False],
        'extra_features': [False], 'cell_neurons': [64, 128, 256, 512]
    }),
)


BEST_SKLEARN_MODELS = (
    (SGDRegressorModel, {
        'max_data': None, 'data_type': 'step', 'norm_func': 'standardize', 'loss': 'huber', 'penalty': 'l1'
    }),
    (KernelRidgeModel, {
        'max_data': 50_000, 'data_type': 'step', 'norm_func': 'linear'
    }),
    (ElasticNetModel, {
        'max_data': None, 'data_type': 'step', 'norm_func': 'standardize'
    }),
    (PassiveAggressiveModel, {
        'max_data': None, 'data_type': 'step', 'norm_func': 'noop'
    }),
    (RandomForestModel, {
        'n_estimators': 300, 'max_depth': None, 'max_data': None, 'data_type': 'step', 'norm_func': 'standardize'
    }),
    (GeneralLinearModel, {
        'max_data': 50_000, 'data_type': 'step', 'norm_func': 'standardize', 'power': 0
    }),
    (AdaBoostModel, {
        'max_depth': None, 'max_data': 50_000, 'data_type': 'step', 'norm_func': 'standardize'
    }),
)

BEST_NN_MODELS = (
    (MLPModel, {
        'epochs': 75, 'data_type': 'step', 'norm_func': 'tanh', 'batch_norm': True,
        'dp': 0.2, 'act': 'tanh', 'include_lanes': True, 'extra_features': True, 'hidden_layers': 6
    }),
    (RNNModel, {
        'epochs': 75, 'data_type': 'step', 'norm_func': 'tanh', 'batch_norm': True,
        'rnn_type': 'lstm', 'dp': 0.2, 'act': 'tanh', 'include_lanes': False,
        'extra_features': False, 'cell_neurons': 512, 'hidden_layers': 5
    })
)


def test_models(models='initial_sklearn', start=0):
    models_name = models
    path = DATA_PATH + "/%s_test_models.pkl" % models_name

    best = False
    if models == 'initial_sklearn':
        models = INITIAL_SKLEARN_MODELS
    elif models == 'secondary_sklearn':
        models = SECONDARY_SKLEARN_MODELS
    elif models == 'initial_nn':
        models = INITIAL_NN_MODELS
    elif models == 'secondary_nn':
        models = SECONDARY_NN_MODELS
    elif models == 'final_nn':
        models = FINAL_NN_MODELS
    elif models == "best_sklearn":
        best = True
        models = BEST_SKLEARN_MODELS
    elif models == "best_nn":
        best = True
        models = BEST_NN_MODELS
    else:
        raise ValueError("Unknown models: %s" % models)

    outputs = []

    idx = 0
    for model, kwargs in models:
        if idx < start:
            idx += 1
            continue
        for tkwargs in _get_all_kwargs(kwargs):
            print("%s: Running model %s with kwargs: %s" % (datetime.now().strftime("%H:%M:%S"), model.__name__, tkwargs))
            try:
                tkwargs['callback'] = best
                tkwargs['save_model'] = best
                curr_model = model(**tkwargs)
                curr_model.train()
                rmses = [curr_model.rmse, curr_model.real_rmse]
            except Exception as e:
                print("ERROR on class %s: %s" % (model.__name__, e))
                rmses = [[_M], [_M]]
            outputs.append([model.__name__, {k: v for k, v in tkwargs.items()}, rmses])
            with open(path, 'wb') as f:
                pickle.dump(outputs, f)


def _get_all_kwargs(kwargs):
    # Convert everything to lists
    for k, v in kwargs.items():
        if not isinstance(v, list):
            if isinstance(v, tuple):
                kwargs[k] = [x for x in v]
            else:
                kwargs[k] = [v]

    # Add all the lists together into one list
    keys = list(kwargs.keys())
    for l in product(*[kwargs[k] for k in keys]):
        yield {k: v for k, v in zip(keys, l)}
