from ForecastModels.ForecastModel import *


class MLPModel(ForecastModel):
    def __init__(self, redo=False, norm_func=None, y_output='full', hidden_layers=3, batch_norm=False, act='elu',
                 dp=None, loss='mse', optimizer='nadam', epochs=10, batch_size=256, final_act=None,
                 kernel_regularizer=None, bias_regularizer='same', l1_val=1e-3, l2_val=1e-3, layer_sizes=None,
                 batch_norm_before_non_linearity=True, batch_norm_last_layer=False, print_model_summary=True):

        _reg = lambda r: l1(l1_val) if r == 'l1' else l2(l2_val) if r == 'l2' else l1_l2(l1=l1_val, l2=l2_val) \
            if r == 'l1_l2' else None
        if bias_regularizer == 'same':
            bias_regularizer = kernel_regularizer
        kernel_regularizer, bias_regularizer = _reg(kernel_regularizer), _reg(bias_regularizer)

        super().__init__(redo, norm_func, y_output, n_layers=hidden_layers, batch_norm=batch_norm, act=act,
                         loss=loss, optim=optimizer, bnbl=batch_norm_before_non_linearity, bnll=batch_norm_last_layer,
                         pms=print_model_summary, dp=dp, final_act=final_act, kr=kernel_regularizer,
                         br=bias_regularizer, layer_sizes=layer_sizes)
        self.epochs = epochs
        self.batch_size = batch_size

    def _gen_model(self, **kwargs):
        in_dim = self.data['train_x'].shape[1]
        out_dim = self.data['train_y'].shape[1]
        n_layers = kwargs['n_layers']

        kwargs['final_act'] = kwargs['act'] if kwargs['final_act'] is None else kwargs['final_act']

        ki = 'random_normal' if kwargs['act'] != 'selu' else keras.initializers.RandomNormal(stddev=np.sqrt(1.0 / in_dim))

        if kwargs['layer_sizes'] is not None:
            s = kwargs['layer_sizes'] + [out_dim]
            n_layers = len(s) - 1
        else:
            s = np.geomspace(in_dim, out_dim, num=n_layers + 1, endpoint=True).astype(int)

        model = Sequential()

        for i in range(n_layers + 1):
            bn = (kwargs['bnll'] and i == n_layers - 1) or i < n_layers - 1

            model.add(Dense(s[i], input_dim=in_dim if i == 0 else None, kernel_initializer=ki, bias_initializer='ones',
                            kernel_regularizer=kwargs['kr'], bias_regularizer=kwargs['br']))
            if kwargs['batch_norm'] and kwargs['bnbl'] and bn:
                model.add(BatchNormalization())
            model.add(Activation(activation=kwargs['final_act' if i == n_layers - 1 else 'act']))
            if kwargs['batch_norm'] and not kwargs['bnbl'] and bn:
                model.add(BatchNormalization())
            if kwargs['dp'] is not None and kwargs['dp'] > 0 and i < n_layers - 1:
                model.add(Dropout(kwargs['dp']))

        model.compile(optimizer=kwargs['optim'], loss=kwargs['loss'], metrics=['mse'])

        if kwargs['pms']:
            model.summary()

        return model

    def _train_model(self):
        self.model.fit(self.data['train_x'], self.data['train_y'], epochs=self.epochs, batch_size=self.batch_size)


class RNNModel(ForecastModel):
    def __init__(self, redo=False, norm_func=None, y_output='full', epochs=10, batch_size=256):
        super().__init__(redo, norm_func, y_output)
        self.epochs = epochs
        self.batch_size = batch_size

    def _conv_data(self):
        """
        For converting the data into an RNN acceptable format
        """
        xp = self.data['train_x'][:19*60*2].reshape([19, 60, 2])
        xv = self.data['train_x'][19*60*2:19*60*4].reshape([19, 60, 2])
        xl = self.data['train_x'][-2 * MAX_LANES * 2:]


    def _gen_model(self, **kwargs):
        # Reformat the data here first
        in_dim = self.data['train_x'].shape[1]
        out_dim = self.data['train_y'].shape[1]
        n_layers = kwargs['n_layers']

        s = np.geomspace(in_dim, out_dim, num=n_layers + 1, endpoint=True).astype(int)

        model = Sequential()

        for i in range(n_layers + 1):
            bn = (kwargs['bnll'] and i == n_layers - 1) or i < n_layers - 1

            model.add(Dense(s[i], input_dim=in_dim if i == 0 else None, kernel_initializer='random_normal'))
            if kwargs['batch_norm'] and kwargs['bnbl'] and bn:
                model.add(BatchNormalization())
            model.add(Activation(activation=kwargs['act']))
            if kwargs['batch_norm'] and not kwargs['bnbl'] and bn:
                model.add(BatchNormalization())
            if kwargs['dp'] is not None and kwargs['dp'] > 0 and i < n_layers - 1:
                model.add(Dropout(kwargs['dp']))

        model.compile(optimizer=kwargs['optim'], loss=kwargs['loss'], metrics=['mse'])

        if kwargs['pms']:
            model.summary()

        return model

    def _train_model(self):
        self.model.fit(self.data['train_x'], self.data['train_y'], epochs=self.epochs, batch_size=self.batch_size)
