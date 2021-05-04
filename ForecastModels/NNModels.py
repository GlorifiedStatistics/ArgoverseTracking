from ForecastModels.ForecastModel import *
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Dropout, SimpleRNN, LSTM, GRU
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import Callback


class NNCallback(Callback):
    def __init__(self, nn_model):
        super().__init__()
        self.nn_model = nn_model

    def on_epoch_end(self, epoch, logs=None):
        self.nn_model.pred_metrics()


class MLPModel(ForecastModel):
    def __init__(self, hidden_layers=3, batch_norm=False, act='elu', dp=None, loss='mse', optimizer='nadam', epochs=10,
                 batch_size=256, kernel_regularizer=None, bias_regularizer='same', l1_val=1e-3, l2_val=1e-3,
                 layer_sizes=None, callback=False, **kwargs):

        # Act can be a list with first element the inner layers activation, and second the final layer
        if isinstance(act, (list, tuple)):
            act = act[0]
            final_act = act[1]
        else:
            act = act
            final_act = None

        _reg = lambda r: l1(l1_val) if r == 'l1' else l2(l2_val) if r == 'l2' else l1_l2(l1=l1_val, l2=l2_val) \
            if r == 'l1_l2' else None
        if bias_regularizer == 'same':
            bias_regularizer = kernel_regularizer
        kernel_regularizer, bias_regularizer = _reg(kernel_regularizer), _reg(bias_regularizer)

        super().__init__(hidden_layers=hidden_layers, batch_norm=batch_norm, act=act,
                         loss=loss, optim=optimizer, dp=dp, final_act=final_act, kr=kernel_regularizer,
                         br=bias_regularizer, layer_sizes=layer_sizes, **kwargs)
        self.epochs = epochs
        self.batch_size = batch_size
        self._callback = callback

    @staticmethod
    def _add_layers(model, in_dim, out_dim, **kwargs):
        n_layers = kwargs['hidden_layers']

        kwargs['final_act'] = kwargs['act'] if kwargs['final_act'] is None else kwargs['final_act']

        if kwargs['layer_sizes'] is not None:
            s = kwargs['layer_sizes'] + [out_dim]
            n_layers = len(s) - 1
        else:
            s = np.geomspace(in_dim, out_dim, num=n_layers + 1, endpoint=True).astype(int)

        for i in range(n_layers + 1):
            model.add(Dense(s[i], input_dim=in_dim if i == 0 else None, kernel_regularizer=kwargs['kr'],
                            bias_regularizer=kwargs['br']))
            if kwargs['batch_norm'] and i < n_layers - 1:
                model.add(BatchNormalization())
            model.add(Activation(activation=kwargs['final_act' if i == n_layers - 1 else 'act']))
            if kwargs['dp'] is not None and kwargs['dp'] > 0 and i < n_layers - 1:
                model.add(Dropout(kwargs['dp']))

        model.compile(optimizer=kwargs['optim'], loss=kwargs['loss'], metrics=['mse'])
        model.summary()

    def _gen_model(self, **kwargs):
        model = Sequential()
        self._add_layers(model, self.data['train_x'].shape[1], self.data['train_y'].shape[1], **kwargs)
        return model

    def _train_model(self):
        self.model.fit(self.data['train_x'], self.data['train_y'], epochs=self.epochs, batch_size=self.batch_size,
                       callbacks=[NNCallback(self)] if self._callback else None)

    def save(self):
        path = self._model_path()[:-1 - len(MODEL_EXT)]
        print("Saving NN model to: '%s'" % path)
        self.model.save(path)

    def load(self):
        path = self._model_path()[:-1 - len(MODEL_EXT)]
        print("Loading NN model from: '%s'" % path)
        return load_model(self._model_path()[:1 + len(MODEL_EXT)])


class RNNModel(MLPModel):
    def __init__(self, cell_neurons=256, rnn_act='tanh', rnn_type='LSTM', **kwargs):
        super().__init__(cell_neurons=cell_neurons, rnn_act=rnn_act, rnn_type=rnn_type, **kwargs)

        # Convert data into rnn-acceptable
        trx = self.data['train_x']
        tx = self.data['test_x']
        vx = self.data['val_x']
        lanes = 4 * MAX_LANES if self._include_lanes else 0
        extra = 1 if self._extra_features else 0

        new_train_x = np.empty([len(trx), 19, (4 + extra) * 60 + lanes])
        new_test_x = np.empty([len(tx), 19, (4 + extra) * 60 + lanes])
        new_val_x = np.empty([len(vx), 19, (4 + extra) * 60 + lanes])

        for old_d, new_d in zip([trx, tx, vx], [new_train_x, new_test_x, new_val_x]):
            s, e, a, i = 0, 19 * 60 * 2, 0, 60 * 2
            new_d[:, :, a:i] = old_d[:, s:e].reshape([-1, 19, 60 * 2])

            s, e, a, i = e, e + 19 * 60 * 2, i, i + 60 * 2
            new_d[:, :, a:i] = old_d[:, s:e].reshape([-1, 19, 60 * 2])

            if self._include_lanes:
                s, e, a, i = e, e + lanes, i, i + lanes
                new_d[:, :, a:i] = np.repeat(old_d[:, s:e], 19, axis=0).reshape([-1, 19, 4 * MAX_LANES])

            if self._extra_features:
                s, e, a, i = e, e + 19 * 60, i, i + 60
                new_d[:, :, a:i] = old_d[:, s:e].reshape([-1, 19, 60])

        self.data['train_x'] = new_train_x.copy()
        self.data['test_x'] = new_test_x.copy()
        self.data['val_x'] = new_val_x.copy()

    def _gen_model(self, **kwargs):
        kwargs['rnn_type'] = kwargs['rnn_type'].lower()
        if kwargs['rnn_type'] in ['rnn', 'simple']:
            rnn_type = SimpleRNN
        elif kwargs['rnn_type'] in ['lstm']:
            rnn_type = LSTM
        elif kwargs['rnn_type'] in ['gru']:
            rnn_type = GRU
        else:
            raise ValueError("Unknown rnn_type: %s" % kwargs['rnn_type'])

        d_size = 60 * 4
        d_size += 4 * MAX_LANES if self._include_lanes else 0
        d_size += 60 if self._extra_features else 0

        # Set up the RNN layers
        model = Sequential()
        model.add(rnn_type(kwargs['cell_neurons'], activation=kwargs['rnn_act'],
                           input_shape=(19, d_size)))

        self._add_layers(model, kwargs['cell_neurons'], 60, **kwargs)

        return model