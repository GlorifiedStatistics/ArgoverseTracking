from ForecastModels.ForecastModel import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.regularizers import l1, l2, l1_l2
from attention import Attention as KerasAttention


class NNCallback(Callback):
    def __init__(self, nn_model):
        super().__init__()
        self.nn_model = nn_model

    def on_epoch_end(self, epoch, logs=None):
        self.nn_model.pred_metrics()


class MLPModel(ForecastModel):
    def __init__(self, hidden_layers=3, batch_norm=True, act='elu', dp=None, loss='mse', optimizer='nadam', epochs=10,
                 batch_size=256, callback=False, layer_sizes=None, regularizer=None, **kwargs):

        self.hidden_layers = hidden_layers
        self.batch_norm = batch_norm
        self.dp = dp
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.callback = callback
        self.layer_sizes = layer_sizes
        self.regularizer = l1 if regularizer in ['l1'] else l2 if regularizer in ['l2'] else l1_l2 if regularizer \
            in ['l1_l2'] else regularizer

        # Act can be a list with first element the inner layers activation, and second the final layer
        if isinstance(act, (list, tuple)):
            self.act = act[0]
            self.final_act = act[1]
        else:
            self.act = act
            self.final_act = None

        super().__init__(**kwargs)

    def _gen_model(self):
        input_layer = Input(shape=self.train_x.shape[1])
        output_layer = self._add_layers(input_layer, self.train_x.shape[1], self.train_y.shape[1])
        return self._finalize_model(input_layer, output_layer)

    def _train_model(self):
        self.model.fit(self.train_x, self.train_y, epochs=self.epochs, batch_size=self.batch_size,
                       callbacks=[NNCallback(self)] if self.callback else None)

    def _finalize_model(self, inputs, outputs):
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=self.optimizer, loss=self.loss)
        model.summary()
        return model

    def save(self):
        path = self._model_path()[:-1 - len(MODEL_EXT)]
        print("Saving NN model to: '%s'" % path)
        self.model.save(path)

    def load(self):
        path = self._model_path()[:-1 - len(MODEL_EXT)]
        print("Loading NN model from: '%s'" % path)
        return load_model(self._model_path()[:1 + len(MODEL_EXT)])

    def _add_layers(self, model, in_dim, out_dim, time_distributed=False):
        """
        Adds some set of Dense layers to the end of the given model
        """
        td = lambda x: TimeDistributed(x) if time_distributed else x

        if self.layer_sizes is not None:
            s = self.layer_sizes + [out_dim]
            n_layers = len(s)
        else:
            s = np.geomspace(in_dim, out_dim, num=self.hidden_layers + 1, endpoint=True).astype(int)
            n_layers = self.hidden_layers + 1

        for i in range(n_layers):
            model = td(Dense(s[i], kernel_regularizer=self.regularizer, bias_regularizer=self.regularizer))(model)
            if self.batch_norm and i < n_layers - 1:
                model = td(BatchNormalization())(model)
            model = td(Activation(activation=self.final_act if i == n_layers - 1 else self.act))(model)
            if self.dp is not None and self.dp > 0 and i < n_layers - 1:
                model = td(Dropout(self.dp))(model)

        return model


class RNNModel(MLPModel):
    def __init__(self, cell_neurons=256, rnn_act='sigmoid', rnn_type='LSTM', rnn_dp=None, rnn_input_dp=None,
                 bidirectional=False, attention=False, **kwargs):

        self.cell_neurons = cell_neurons
        self.rnn_act = rnn_act
        self.rnn_type = rnn_type
        self.rnn_dp = rnn_dp if rnn_dp is not None else 0
        self.rnn_input_dp = rnn_input_dp if rnn_input_dp is not None else 0
        self.bidirectional = bidirectional
        self.attention = attention

        # Must do this first, before the _gen_model() call
        rnn_type = rnn_type.lower()
        self.rnn_type = SimpleRNN if rnn_type in ['rnn', 'simple'] else LSTM if rnn_type in ['lstm'] else \
            GRU if rnn_type in ['gru'] else None

        super().__init__(**kwargs)

    def _conv_data(self, data):
        """
        Overwrites data in the dictionary data
        """
        # Convert data into rnn-acceptable
        trx = data['train_x']
        tx = data['test_x']
        vx = data['val_x']
        lanes = 4 * MAX_LANES
        extra = 1 if self.extra_features else 0

        new_train_x = np.empty([len(trx), 19, (4 + extra) * 60])
        new_train_x_lanes = np.empty([len(trx), lanes])
        new_test_x = np.empty([len(tx), 19, (4 + extra) * 60])
        new_test_x_lanes = np.empty([len(tx), lanes])
        new_val_x = np.empty([len(vx), 19, (4 + extra) * 60])
        new_val_x_lanes = np.empty([len(vx), lanes])

        for old_d, new_d, new_d_lanes in zip([trx, tx, vx], [new_train_x, new_test_x, new_val_x],
                                             [new_train_x_lanes, new_test_x_lanes, new_val_x_lanes]):
            s, e, a, i = 0, 19 * 60 * 2, 0, 60 * 2
            new_d[:, :, a:i] = old_d[:, s:e].reshape([-1, 19, 60 * 2])

            s, e, a, i = e, e + 19 * 60 * 2, i, i + 60 * 2
            new_d[:, :, a:i] = old_d[:, s:e].reshape([-1, 19, 60 * 2])

            if self.include_lanes:
                s, e = e, e + lanes
                new_d_lanes[:, :] = old_d[:, s:e]

            if self.extra_features:
                s, e, a, i = e, e + 19 * 60, i, i + 60
                new_d[:, :, a:i] = old_d[:, s:e].reshape([-1, 19, 60])

        data['train_x'] = [new_train_x.copy(), new_train_x_lanes.copy()] if self.include_lanes else new_train_x.copy()
        data['test_x'] = [new_test_x.copy(), new_test_x_lanes.copy()] if self.include_lanes else new_test_x.copy()
        data['val_x'] = [new_val_x.copy(), new_val_x_lanes.copy()] if self.include_lanes else new_val_x.copy()

        return data

    def _rnn_layer(self, layer=None, seq=False, state=False, attention=None):
        attention = self.attention if attention is None else attention
        seq = True if attention else seq
        l = self.rnn_type(self.cell_neurons, activation=self.act, recurrent_activation=self.rnn_act,
                          dropout=self.rnn_input_dp, recurrent_dropout=self.rnn_dp, return_sequences=seq,
                          return_state=state)
        l = Bidirectional(l) if self.bidirectional else l
        return l if layer is None else l(layer)

    def _input_shape(self):
        return [[19, 60 * (5 if self.extra_features else 4)]] + ([[4 * MAX_LANES]] if self.include_lanes else [])

    def _gen_model(self):
        # Set up the RNN layers
        lane_layer = Input(shape=self._input_shape()[1]) if self.include_lanes else None
        input_layer = Input(shape=self._input_shape()[0])
        inputs = [input_layer, lane_layer] if self.include_lanes else input_layer

        rnn_layer = self._rnn_layer(input_layer)
        att = Flatten()(KerasAttention()(rnn_layer)) if self.attention else rnn_layer
        concat = concatenate([att, lane_layer]) if self.include_lanes else att
        output_layer = self._add_layers(concat, self.cell_neurons, 60)

        return self._finalize_model(inputs, output_layer)
