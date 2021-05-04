from utils.Utils import *
from Argoverse.Datasets import load_dataset
from Argoverse.NormFuncs import get_norm_funcs


class ForecastModel:
    def __init__(self, norm_func, max_data=None, data_type='step', data_and_stats=None, include_lanes=True,
                 extra_features=False, y_norm_func=None, load_model=False, save_model=False, **kwargs):
        self.data_type = data_type
        if data_and_stats is None:
            self.data, self.stats = load_dataset(norm_func=norm_func, max_data=max_data, data_type=data_type,
                                                 include_lanes=include_lanes, extra_features=extra_features,
                                                 y_norm_func=y_norm_func, **kwargs)
        else:
            self.data, self.stats = data_and_stats

        self.name = "%s_%s_%s_" % (norm_func, data_type, y_norm_func) + ("all" if max_data is None else "%d" % max_data) + "_"

        self._include_lanes = include_lanes
        self._extra_features = extra_features

        if load_model:
            self.model = "Load"
        else:
            self.model = self._gen_model(**kwargs)
        self.inv_y_norm_func = get_norm_funcs(y_norm_func)[1] if y_norm_func else get_norm_funcs(norm_func)[1]
        self._trained = False
        self.rmse = []
        self.real_rmse = []
        self.best_real_rmse = 100000000000
        self.best_val_pred = None
        self.save_model = save_model

    def _model_path(self):
        return os.path.join(MODELS_PATH, self.__class__.__name__ + "_" + self.name + MODEL_EXT)

    def _gen_model(self, **kwargs):
        pass

    def _train_model(self):
        pass

    def _predict(self, X):
        """
        Returns non-normalized predictions
        """
        if isinstance(self.model, str):
            self.model = self.load()
        return self.model.predict(X)

    def pred_metrics(self):
        rmse, real_rmse, _ = pred_metrics(self._predict(self.data['test_x']), self.data, self.stats,
                                          self.inv_y_norm_func, self.data_type)
        print("Model had RMSE: %f, and real-world RMSE: %f" % (rmse, real_rmse))

        self.rmse.append(rmse)
        self.real_rmse.append(real_rmse)

        if real_rmse < self.best_real_rmse:
            print("New best RMSE!, doing val...")
            self.best_real_rmse = real_rmse
            self.best_val_pred = unpack_y(self.inv_y_norm_func(self._predict(self.data['val_x']), self.stats),
                                          self.data, val=True)
            if self.save_model:
                print("Saving model...")
                self.save()

    def other_metrics(self):
        """
        :return: the rmse and real_rmse, along with other metrics defined in other_metrics() outside this class
        """
        rmse, real_rmse, pred = pred_metrics(self._predict(self.data['test_x']), self.data, self.stats,
                                             self.inv_y_norm_func, self.data_type)
        y_true = unpack_y(self.inv_y_norm_func(self.data['test_y'], self.stats), self.data, test=True)
        mr, fde, ade = other_metrics(pred, y_true)

        return rmse, real_rmse, mr, fde, ade

    def train(self):
        print("Training %s model on data of shape %s..." % (self.__class__.__name__, self.data['train_x'].shape))
        self._train_model()
        self._trained = True
        print("All done training, testing...")
        self.pred_metrics()

    def compute_val(self):
        if not self._trained:
            raise Exception("Model has not yet been trained!")

        if self.best_val_pred is None:
            print("No previous best RMSE, doing val...")
            self.best_val_pred = unpack_y(self.inv_y_norm_func(self._predict(self.data['val_x']), self.stats),
                                          self.data, val=True)
        name = self.__class__.__name__ + "_" + self.name
        save_predictions(self.best_val_pred, name, self.data['val_labels'])

    def test_predictions(self):
        """
        Returns the y_pred and y_true of this model on the test values
        """
        _, _, pred = pred_metrics(self._predict(self.data['test_x']), self.data, self.stats,
                                  self.inv_y_norm_func, self.data_type)
        x = self.inv_y_norm_func(self.data['test_x'], self.stats)[:, :19 * 60 * 2]
        x = x.reshape([x.shape[0], 19, 60, 2])
        x = unpack_y(x[:, :, 0, :].reshape([x.shape[0], -1]), self.data, test=True)
        return x, pred, unpack_y(self.data['test_y'], self.data)

    def save(self):
        raise NotImplementedError()

    def load(self):
        raise NotImplementedError()


def pred_metrics(pred, data, stats, inv_norm_func, data_type):
    # Make the metrics for model output
    y = data['test_y']
    rmse = (np.sum((pred - y) ** 2) / (y.shape[0] * y.shape[1])) ** 0.5

    # Undo the normalization
    y = inv_norm_func(y, stats, data_type=data_type)
    pred = inv_norm_func(pred, stats, data_type=data_type)

    # Shift values into their correct positions
    y = unpack_y(y, data)
    pred = unpack_y(pred, data)

    # Make metrics for real-world output
    real_rmse = (np.sum((pred - y) ** 2) / (y.shape[0] * y.shape[1])) ** 0.5

    return rmse, real_rmse, pred


def unpack_y(y, data, val=False, test=False):
    """
    Converts the given y value or prediction to original coordinates
    :param y: the y value
    :param data: the data
    :param val: whether or not this is on the validation set
    :param test: whether or not this is the test set x
    :return: unpacked y
    """
    val_key = 'val_pred_off' if val else 'test_x_off' if test else 'test_pred_off'
    ret = y.copy().reshape([y.shape[0], -1, 2])
    ret[:, 0, :] += data[val_key]
    return np.cumsum(ret, axis=1).reshape([y.shape[0], -1])


def other_metrics(y_pred, y_true, miss_dist=2):
    """
    Calculates other metrics that seem to be used by Argoverse, so I can see how close (or more likely, far) I am to
    the best of the best
    - miss_rate: the percent of predictions that end more than miss_dist meters away from the ground truth
    - fde: final displacement error: the average distance from endpoints to ground truth
    - ade: average displacement error: the average distance from ALL points to ground truth
    :return:
    """
    # The euclidean distance between all points in p and y
    def _dist(p, y):
        return np.sum((p - y) ** 2, axis=1) ** 0.5

    # The endpoint displacement
    end_disp = _dist(y_pred[:, -2:], y_true[:, -2:])

    miss_rate = len(np.where(end_disp > miss_dist)) / len(y_pred)
    fde = np.sum(end_disp) / len(y_pred)

    ade = 0
    for i in range(30):
        s = 2 * i
        e = s + 1
        ade += np.sum(_dist(y_pred[:, s:e], y_true[:, s:e])) / len(y_pred)

    return miss_rate, fde, ade / 30
