from utils.Utils import *
from Argoverse.Datasets import load_dataset
from Argoverse.NormFuncs import get_norm_funcs


def _rmse(y_pred, y_true):
    return (np.sum((y_pred - y_true) ** 2) / len(y_true.reshape([-1]))) ** 0.5


def _dist(p, y):
    return np.sum((p - y) ** 2, axis=1) ** 0.5


class ForecastModel:
    def __init__(self, norm_func='noop', max_data=None, data_type='step', data_and_stats=None, include_lanes=True,
                 extra_features=False, y_norm_func=None, load_model=False, save_model=False, **nf_kwargs):
        self.norm_func = norm_func
        self.max_data = max_data
        self.data_type = data_type
        self.include_lanes = include_lanes
        self.extra_features = extra_features
        self.y_norm_func = norm_func if y_norm_func is None else y_norm_func
        self.load_model = load_model
        self.save_model = save_model
        self.nf_kwargs = nf_kwargs
        self._trained = False
        self.rmse = []
        self.real_rmse = []
        self.miss_rate = []
        self.fde = []
        self.ade = []
        self.best_real_rmse = 100000000000
        self.best_val_pred = None
        self.last_test_pred = None

        inv_y_func = get_norm_funcs(y_norm_func)[1] if y_norm_func else get_norm_funcs(norm_func)[1]
        self.inv_y_norm_func = lambda pred: inv_y_func(pred, self.stats, data_type=self.data_type, **nf_kwargs)

        if data_and_stats is None:
            data, self.stats = load_dataset(norm_func=norm_func, max_data=max_data, data_type=data_type,
                                            include_lanes=include_lanes, extra_features=extra_features,
                                            y_norm_func=y_norm_func, **nf_kwargs)
        else:
            data, self.stats = data_and_stats

        data = self._conv_data(data)

        self.train_x = data['train_x']
        self.test_x = data['test_x']
        self.val_x = data['val_x']
        self.train_y = data['train_y']
        self.test_y = data['test_y']
        self.test_x_off = data['test_x_off']
        self.test_pred_off = data['test_pred_off']
        self.val_pred_off = data['val_pred_off']
        self.val_labels = data['val_labels']

        self.name = "%s_%s_%s_" % (norm_func, data_type, y_norm_func) + \
                    ("all" if max_data is None else "%d" % max_data) + "_"

        self.model = "Load" if load_model else self._gen_model()

    def _model_path(self):
        return os.path.join(MODELS_PATH, self.__class__.__name__ + "_" + self.name + MODEL_EXT)

    def _gen_model(self):
        pass

    def _conv_data(self, data):
        return data

    def _train_model(self):
        pass

    def _predict(self, X):
        """
        Returns non-normalized predictions
        """
        if isinstance(self.model, str):
            self.model = self.load()
        return self.model.predict(X)

    def _unroll_y(self, y, val=False, test=False):
        ret = y.copy().reshape([y.shape[0], -1, 2])
        ret[:, 0, :] += self.val_pred_off if val else self.test_x_off if test else self.test_pred_off
        return np.cumsum(ret, axis=1).reshape([y.shape[0], -1])

    def pred_metrics(self):
        # Make the metrics for model output
        pred = self._predict(self.test_x)
        rmse = _rmse(pred, self.test_y)

        # Undo the normalization and unroll
        y = self._unroll_y(self.inv_y_norm_func(self.test_y))
        pred = self._unroll_y(self.inv_y_norm_func(pred))

        # Make metrics for real-world output
        real_rmse = _rmse(pred, y)

        # Do the other metrics
        end_disp = _dist(pred[:, -2:], y[:, -2:])

        miss_rate = len(np.where(end_disp > MISS_DIST)) / pred.shape[0]
        fde = np.sum(end_disp) / pred.shape[0]

        ade = 0
        for i in range(30):
            s = 2 * i
            e = s + 1
            ade += np.sum(_dist(pred[:, s:e], y[:, s:e])) / len(pred)

        ade /= 30

        print("Model had MSE: %f, RMSE: %f, and real-world RMSE: %f" % (rmse ** 2, rmse, real_rmse))

        self.rmse.append(rmse)
        self.real_rmse.append(real_rmse)
        self.miss_rate.append(miss_rate)
        self.fde.append(fde)
        self.ade.append(ade)

        if real_rmse < self.best_real_rmse:
            print("New best RMSE!, doing val...")
            self.best_real_rmse = real_rmse
            self.best_val_pred = self._get_val_pred()
            if self.save_model:
                print("Saving model...")
                self.save()

    def train(self):
        if isinstance(self.train_x, (list, tuple)):
            print("Training %s model on multiple datas of shapes: %s" %
                  (self.__class__.__name__, [a.shape for a in self.train_x]))
        else:
            print("Training %s model on data of shape %s..." % (self.__class__.__name__, self.train_x.shape))
        self._train_model()
        self._trained = True
        print("All done training, testing...")
        self.pred_metrics()

    def _get_val_pred(self):
        return self._unroll_y(self.inv_y_norm_func(self._predict(self.val_x)), val=True)

    def compute_val(self):
        if not self._trained:
            raise Exception("Model has not yet been trained!")

        if self.best_val_pred is None:
            print("No previous best RMSE, doing val...")
            self.best_val_pred = self._get_val_pred()
        name = self.__class__.__name__ + "_" + self.name
        save_predictions(self.best_val_pred, name, self.val_labels)

    def save(self):
        raise NotImplementedError()

    def load(self):
        raise NotImplementedError()
