from ForecastModels.NNModels import *


class RNNEncoderDecoderModel(RNNModel):
    def _gen_model(self):
        class _M(nn.Module):
            def __init__(self, input_size, rnn_size, num_rnn_layers, rnn_dp, bi, act, dp, include_lanes):
                super().__init__()
                input_size, *lane_size = input_size
                self.rnn_size = rnn_size
                self.bi = bi
                self.num_rnn_layers = num_rnn_layers
                self.encoder = nn.LSTM(input_size[1], hidden_size=rnn_size, dropout=rnn_dp,
                                       bidirectional=bi, batch_first=True, num_layers=num_rnn_layers)
                self.decoder = nn.LSTM(rnn_size * (2 if bi else 1) + (4 * MAX_LANES if include_lanes else 0),
                                       hidden_size=rnn_size, dropout=rnn_dp, bidirectional=bi, batch_first=True,
                                       num_layers=num_rnn_layers)
                rs = self.rnn_size * (2 if bi else 1)
                self.ff1 = nn.Linear(rs, 512) if include_lanes else nn.Linear(rs, 512)
                self.ff2 = nn.Linear(512, 256)
                self.ff3 = nn.Linear(256, 60)
                self.bn1 = nn.BatchNorm1d(512)
                self.bn2 = nn.BatchNorm1d(256)
                self.dp = dp
                self.act = act
                self.include_lanes = include_lanes

            def forward(self, *args):
                def _dp(v):
                    return nn.Dropout(self.dp)(v) if self.dp is not None and self.dp > 0 else v

                x, *other = args

                h0 = torch.zeros(self.num_rnn_layers * (2 if self.bi else 1), x.size(0), self.rnn_size).requires_grad_()
                c0 = torch.zeros(self.num_rnn_layers * (2 if self.bi else 1), x.size(0), self.rnn_size).requires_grad_()
                h0, c0 = h0.to(DEVICE), c0.to(DEVICE)

                x, (hn, cn) = self.encoder(x, (h0.detach(), c0.detach()))
                x = x[:, -1, :].contiguous().view(-1, self.rnn_size * (2 if self.bi else 1))

                if self.include_lanes:
                    x = torch.cat((x, other[0]), dim=1)

                s = x.shape[-1]
                x = x.repeat([1, 30]).view([-1, 30, s])

                # Use the states from encoder with gradients including encoder
                x, (hn, cn) = self.decoder(x, (hn, cn))
                x = x[:, -1, :].contiguous().view(-1, self.rnn_size * (2 if self.bi else 1))

                x = _dp(self.act(self.bn1(self.ff1(x))))
                x = _dp(self.act(self.bn2(self.ff2(x))))
                x = self.act(self.ff3(x))

                return x

        return _M(self._input_sizes(), self.rnn_size, self.num_rnn_layers, self.rnn_dp, self.bidirectional, self.act,
                  self.dp, self.include_lanes)
