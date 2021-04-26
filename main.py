from Constants import *
from Datasets import *
from Visualize import *
from Models import *


# data_cleaning_norm_functions()

def test_models(y_output='full'):
    models = {
        'rf_small_noop': RandomForestModel(y_output=y_output, norm_func='noop', max_data=10_000),
        'rf_small_linear': RandomForestModel(y_output=y_output, norm_func='linear', max_data=10_000),

        'rf_big_noop': RandomForestModel(y_output=y_output, norm_func='noop', max_depth=None, n_estimators=300, max_data=100_000),
        'rf_big_linear': RandomForestModel(y_output=y_output, norm_func='linear', max_depth=None, n_estimators=300, max_data=100_000),

        'mlp_plain': MLPModel(y_output=y_output, norm_func='linear', epochs=20, hidden_layers=2),
        'mlp_medium_plain': MLPModel(y_output=y_output, norm_func='linear', epochs=20, hidden_layers=3),
        'mlp_large_plain': MLPModel(y_output=y_output, norm_func='linear', epochs=20, hidden_layers=4),
        'mlp_xl_plain': MLPModel(y_output=y_output, norm_func='linear', epochs=20, hidden_layers=6),
    }

    ret = {}

    for name, model in models.items():
        model.train()
        ret[name] = [model.last_rmse, model.last_real_rmse]
        print("\n\n\"%s\" rmse: %f, real_rmse: %f\n\n" % (name, model.last_rmse, model.last_real_rmse))

    with open("./model_rmses.pkl", "wb") as f:
        pickle.dump(ret, f)

    return ret


#test_models()

model = RandomForestModel(max_depth=20, n_estimators=300, max_data=10_000, norm_func='noop')
#model = MLPModel(norm_func='std_step', batch_norm=True, act='selu', epochs=100, final_act=None, dp=None, hidden_layers=3, batch_size=512)
#model = LinearRegressionModel(norm_func='std_step')
#model = DecisionTreeModel(max_depth=None, max_data=10_000, criterion='poisson')

model.train()
model.compute_val()

print(np.argsort(model.model.feature_importances_))