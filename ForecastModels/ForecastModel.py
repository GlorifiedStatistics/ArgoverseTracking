from Constants import *
from Datasets import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import tensorflow.keras as keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Input, Activation, Dropout
from tensorflow.keras.regularizers import l1, l2, l1_l2


class ForecastModel:
    def __init__(self, redo=False, norm_func=None, y_output='full', include_lanes=True, **kwargs):
        self.data = load_dataset(redo=redo, y_output=y_output, norm_func=norm_func, include_val=True)
        self.include_lanes = include_lanes

        if not include_lanes:
            self.data['train_x'] = self.data['train_x'][:-4 * MAX_LANES]
            self.data['val_x'] = self.data['val_x'][:-4 * MAX_LANES]

        self.model = self._gen_model(**kwargs)
        self.norm_func = NormFuncs.get_norm_funcs(norm_func)[0]
        self.y_output = y_output
        self._trained = False
        self.last_rmse = None
        self.last_real_rmse = None
        self.val_answer = None

    def _gen_model(self, **kwargs):
        pass

    def _train_model(self):
        pass

    def train(self):
        print("Training %s model on data of size %d..." % (self.__class__.__name__, len(self.data['train_x'])))
        self._train_model()
        print("All done training, testing...")
        rmse, real_rmse = pred_metrics(self.model.predict(self.data['test_x']), self.data)
        print("Model had RMSE: %f, and real-world RMSE: %f" % (rmse, real_rmse))

        self._trained = True
        self.last_rmse, self.last_real_rmse = rmse, real_rmse

        return rmse, real_rmse

    def compute_val(self):
        if not self._trained:
            raise Exception("Model has not yet been trained!")

        self.val_answer = unpack_y(self.model.predict(self.data['val_x']), self.data, val=True)
        save_predictions(self.val_answer, self.__class__.__name__, self.norm_func, self.y_output)


def pred_metrics(pred, data):
    # Make the metrics for model output
    y = data['test_y']
    rmse = (np.sum((pred - y) ** 2) / (y.shape[0] * y.shape[1])) ** 0.5

    # Undo the normalization
    size = y.shape
    y = data['inv_norm_func'](y.reshape([size[0], -1, 2])).reshape([-1, size[1]])
    pred = data['inv_norm_func'](pred.reshape([size[0], -1, 2])).reshape([-1, size[1]])

    # Shift values into their correct positions
    y = unpack_y(y, data)
    pred = unpack_y(pred, data)

    # Make metrics for real-world output
    real_rmse = (np.sum((pred - y) ** 2) / (y.shape[0] * y.shape[1])) ** 0.5

    return rmse, real_rmse


def unpack_y(y, data, val=False):
    """
    Converts the given y value or prediction to original coordinates
    :param y: the y value
    :param data: the statistics
    :param val: whether or not this is on the validation set
    :return: unpacked y
    """
    val_key = 'val_pred_off' if val else 'test_pred_off'
    # Check for which y_output we are
    num_rows = y.shape[0] if y.shape[1] > 2 else None
    ret = y.copy().reshape([-1, 2])

    # If it's just single_step, it's a bit easier
    if num_rows is None:
        return ret + data[val_key]

    # Otherwise, it's harder
    else:
        ret = ret.reshape([num_rows, 30, 2])
        ret[:, 0, :] += data[val_key]
        return np.cumsum(ret, axis=1).reshape([-1, 60])
