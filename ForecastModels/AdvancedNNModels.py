from ForecastModels.NNModels import *


class RNNEncoderDecoderModel(RNNModel):
    def _gen_model(self):
        # Set up the RNN layers
        lane_input = Input(shape=self._input_shape()[1]) if self.include_lanes else None

        rnn_input = Input(shape=self._input_shape()[0])
        encoder = self._rnn_layer(rnn_input)
        att_enc = Flatten()(KerasAttention()(encoder)) if self.attention else encoder
        concat = concatenate([att_enc, lane_input]) if self.include_lanes else att_enc
        state = RepeatVector(30)(concat)
        decoder = self._rnn_layer(state, seq=True)

        output_layer = Flatten()(self._add_layers(decoder, self.cell_neurons, 2, time_distributed=True))

        return self._finalize_model([rnn_input, lane_input] if self.include_lanes else rnn_input, output_layer)


class Seq2SeqModel(RNNModel):
    def __init__(self, **kwargs):
        kwargs['include_lanes'] = False
        super().__init__(**kwargs)

        # Insert a x,y column of zeros for initial input to decoder, and only go up to penultimate time step

    def _conv_data(self, data):
        data = super()._conv_data(data)
        train_y_decoder = np.append(np.zeros([data['train_y'].shape[0], 2]), data['train_y'][:, :-2], axis=1)
        data['train_x'] = [data['train_x'], train_y_decoder.reshape([-1, 30, 2])]
        return data

    def _gen_model(self):
        """
        Code adapted from:
        https://machinelearningmastery.com/define-encoder-decoder-sequence-sequence-model-neural-machine-translation-keras/
        """
        # Set up the RNN layers

        encoder_input = Input(shape=self._input_shape()[0])
        encoder, *encoder_states = self._rnn_layer(state=True)(encoder_input)

        decoder_inputs = Input(shape=(None, 2))
        decoder_rnn_layer = self._rnn_layer(seq=True, state=True)
        decoder_outputs, *_ = decoder_rnn_layer(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(2, activation=self.act)
        decoder_outputs = Flatten()(decoder_dense(decoder_outputs))

        model = self._finalize_model([encoder_input, decoder_inputs], decoder_outputs)

        self.encoder_model = Model(encoder_input, encoder_states)
        # define decoder inference model
        decoder_states_inputs = [Input(shape=(self.cell_neurons,)), Input(shape=(self.cell_neurons,))]
        decoder_outputs, *decoder_states = decoder_rnn_layer(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

        return model

    def _predict(self, X):
        if isinstance(self.model, str):
            self.model = self.load()

        curr_states = self.encoder_model.predict(X)
        target_seq = np.zeros((len(X), 1, 2))

        model_output = np.zeros((len(X), 60))
        for i in range(30):
            pred, *curr_states = self.decoder_model.predict([target_seq] + curr_states)

            # Update the target sequence and current states
            target_seq = pred.copy()

            model_output[:, i * 2: (i + 1) * 2] = pred.reshape([-1, 2])

        return model_output

