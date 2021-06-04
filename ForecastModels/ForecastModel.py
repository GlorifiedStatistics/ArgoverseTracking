from utils.Utils import *
from Argoverse.Datasets import load_dataset
from Argoverse.NormFuncs import get_norm_funcs
from sklearn.model_selection import train_test_split as tts

_UNUSED_KEYS = ['agent_pos', 'scene_idx']


def _rmse(y_pred, y_true):
    return (np.sum((y_pred - y_true) ** 2) / len(y_true.reshape([-1]))) ** 0.5


def _dist(p, y):
    return np.sum((p - y) ** 2, axis=1) ** 0.5


class ForecastModel:
    def __init__(self, norm_func=None, max_data=None, data=None, load_model=False, save_model=False,
                 v=True, s=False, lanes=False, all_data=False, y_norm_func=None, redo=False, **kwargs):
        if all_data:
            v = s = lanes = True

        name_strings = [
            self.__class__.__name__,
            'all' if max_data is None else str(max_data),
            str(norm_func), str(v), str(s), str(lanes),
        ]
        self.name = "_".join(name_strings)

        self.norm_func = get_norm_funcs(norm_func)
        self.y_norm_func = get_norm_funcs(y_norm_func) if y_norm_func is not None else self.norm_func
        self.max_data = max_data

        self.load_model = load_model
        self.save_model = save_model
        self._trained = load_model
        self.rmse = []
        self.real_rmse = []
        self.miss_rate = []
        self.fde = []
        self.ade = []
        self.best_real_rmse = 100000000000
        self.best_val_pred = None
        self.last_test_pred = None

        if data is None:
            data = load_dataset(p=True, v=v, lanes=lanes, s=s, max_data=max_data, fill_data=True, flatten=True,
                                agent_pos=True, redo=redo)

        train_agent_pos = data[0]['agent_pos']
        self.val_agent_pos = data[1]['agent_pos']
        self.val_labels = data[1]['scene_idx']

        # Normalize the data before _conv_data()
        dataset_stats = {}
        for i in range(2):
            for k in [k for k in list(data[0].keys()) if k not in _UNUSED_KEYS]:
                if k == 'p' and i == 0:
                    y_start = 19 * 60 * 2
                    dataset_stats['p1'] = kwargs if i == 0 else dataset_stats['p1']
                    dataset_stats['p2'] = kwargs if i == 0 else dataset_stats['p2']

                    y_data = data[i][k][:, y_start:].reshape([-1, 30, 60, 2])[:, :, 0, :].reshape([-1, 60]).copy()
                    _, self.inv_norm_kwargs = self.y_norm_func(y_data, **kwargs)

                    data[i][k][:, :y_start], dataset_stats['p1'] = self.norm_func(data[i][k][:, :y_start],
                                                                                  **dataset_stats['p1'])
                    data[i][k][:, y_start:], dataset_stats['p2'] = self.y_norm_func(data[i][k][:, y_start:],
                                                                                    **dataset_stats['p2'])
                else:
                    dataset_stats[k] = kwargs if i == 0 else dataset_stats['p1' if k == 'p' else k]
                    data[i][k], dataset_stats[k] = self.norm_func(data[i][k], **dataset_stats)

        self.inv_norm_kwargs['inv'] = True

        data = self._conv_data(data)

        # In case there are multiple arrays for training data
        if isinstance(data[0], (list, tuple)):
            datas = data[0] + [data[1], train_agent_pos]
            #*xs, self.train_y, self.test_y, _, self.test_agent_pos = tts(*datas, test_size=0.2, random_state=73254)
            tx, tsx, tl, tsl, self.train_y, self.test_y, _, self.test_agent_pos = tts(*datas, test_size=0.2, random_state=73254)
            self.train_x = [tx, tl]
            self.test_x = [tsx, tsl]
            #self.train_x, self.test_x = xs[::2], xs[1::2]
        else:
            self.train_x, self.test_x, self.train_y, self.test_y, _, self.test_agent_pos = \
                tts(data[0], data[1], train_agent_pos, test_size=0.2)
        self.val_x = data[2]

        unnorm = self.y_norm_func(self.train_y[0], **self.inv_norm_kwargs)

        self.model = self.load() if load_model else self._gen_model()

    def _model_path(self):
        return os.path.join(MODELS_PATH, self.name + MODEL_EXT)

    def _gen_model(self):
        pass

    def _conv_data(self, data):
        """
        Should return x, y, val_x
        """
        y_start = -30*60*2
        y = data[0]['p'][:, y_start:].reshape([-1, 30, 60, 2])[:, :, 0, :].reshape([-1, 30 * 2])

        for k in [k for k in ['p', 'v', 's'] if k in data[0]]:
            data[0][k] = data[0][k][:, :y_start]

        for i in range(2):
            data[i] = np.concatenate([data[i][k] for k in list(data[i].keys()) if k not in _UNUSED_KEYS], axis=1)

        return data[0], y, data[1]

    def _train_model(self):
        self.model.fit(self.train_x, self.train_y)

    def _predict(self, X):
        """
        Returns non-normalized predictions
        """
        return self.model.predict(X)

    def pred_metrics(self):
        # Make the metrics for model output
        pred = self._predict(self.test_x)
        rmse = _rmse(pred, self.test_y)

        # Undo the normalization and unroll
        y = self.norm_func(self.test_y, **self.inv_norm_kwargs) + np.tile(self.test_agent_pos, 30)
        pred = self.norm_func(pred, **self.inv_norm_kwargs) + np.tile(self.test_agent_pos, 30)

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
        return self.norm_func(self._predict(self.val_x), **self.inv_norm_kwargs) + np.tile(self.val_agent_pos, 30)

    def compute_val(self):
        if not self._trained:
            raise Exception("Model has not yet been trained!")

        if self.best_val_pred is None:
            print("No previous best RMSE, doing val...")
            self.best_val_pred = self._get_val_pred()
        save_predictions(self.best_val_pred, self.name, self.val_labels)

    def save(self):
        raise NotImplementedError()

    def load(self):
        raise NotImplementedError()
