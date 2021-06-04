from ForecastModels.PytorchModels.MLP import *


class _M(nn.Module):
    def __init__(self, input_size, rnn_size, num_rnn_layers, rnn_dp, bi, act, dp, include_lanes):
        super().__init__()
        input_size, lane_size = input_size
        self.rnn_size = rnn_size
        self.bi = bi
        self.num_rnn_layers = num_rnn_layers
        self.dp = dp
        self.act = act
        self.include_lanes = include_lanes

        self.lstm = nn.LSTM(input_size[1] + (256 if include_lanes else 0), hidden_size=rnn_size, dropout=rnn_dp,
                            bidirectional=bi, batch_first=True, num_layers=num_rnn_layers)
        self.bn_in = nn.BatchNorm1d(input_size[0])

        if include_lanes:
            self.lane1 = nn.Linear(lane_size[0], 2048)
            self.lane2 = nn.Linear(2048, 1024)
            self.lane3 = nn.Linear(1024, 256)
            self.bn_lane1 = nn.BatchNorm1d(2048)
            self.bn_lane2 = nn.BatchNorm1d(1024)
            self.bn_lane3 = nn.BatchNorm1d(256)

        self.ff1 = nn.Linear(self.rnn_size * (2 if bi else 1) + (256 if include_lanes else 0), 512)
        self.ff2 = nn.Linear(512, 256)
        self.ff3 = nn.Linear(256, 60)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, *args):
        def _dp(v):
            return nn.Dropout(self.dp)(v) if self.dp is not None and self.dp > 0 else v

        x, *other = args
        x = self.bn_in(x)

        if self.include_lanes:
            lane = _dp(self.act(self.bn_lane1(self.lane1(other[0]))))
            lane = _dp(self.act(self.bn_lane2(self.lane2(lane))))
            lane = _dp(self.act(self.bn_lane3(self.lane3(lane))))
            replane = lane.unsqueeze(1).repeat(1, x.size(1), 1)
            x = torch.cat((x, replane), dim=2)

        h0 = torch.zeros(self.num_rnn_layers * (2 if self.bi else 1), x.size(0), self.rnn_size).requires_grad_()
        c0 = torch.zeros(self.num_rnn_layers * (2 if self.bi else 1), x.size(0), self.rnn_size).requires_grad_()
        h0, c0 = h0.to(DEVICE), c0.to(DEVICE)

        x, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        x = x[:, -1, :].contiguous().view(-1, self.rnn_size * (2 if self.bi else 1))

        if self.include_lanes:
            x = torch.cat((x, lane), dim=1)

        x = _dp(self.act(self.bn1(self.ff1(x))))
        x = _dp(self.act(self.bn2(self.ff2(x))))
        x = self.act(self.ff3(x))

        return x


class BasicRNNModel(MLPModel):
    def __init__(self, rnn_size=256, num_rnn_layers=1, rnn_dp=None, bidirectional=False, **kwargs):

        self.rnn_size = rnn_size
        self.rnn_dp = rnn_dp if rnn_dp is not None else 0
        self.bidirectional = bidirectional
        self.num_rnn_layers = num_rnn_layers

        super().__init__(**kwargs)

    def _conv_data(self, data):
        self.include_lanes = 'lane' in data[0]

        # Get the y and xp
        y_start = -30*60*2
        y = data[0]['p'][:, y_start:].reshape([-1, 30, 60, 2])[:, :, 0, :].reshape([-1, 30 * 2])

        for k in [k for k in ['p', 'v', 's'] if k in data[0]]:
            data[0][k] = data[0][k][:, :y_start]

        for i in range(2):
            # Get all of the non-lane data
            non_lane = [data[i][k].reshape([len(data[i][k]), 19, -1]) for k in ['p', 'v', 's'] if k in data[i]]
            non_lane_data = np.concatenate(non_lane, axis=2)

            data[i] = [non_lane_data, data[i]['lane']] if self.include_lanes else non_lane_data

        self.non_lane_shape = data[0][0].shape
        self.lane_shape = data[0][1].shape if self.include_lanes else None

        return data[0], y, data[1]

    def _gen_model(self):
        input_size = [self.non_lane_shape[1:], self.lane_shape[1:]]
        return _M(input_size, self.rnn_size, self.num_rnn_layers, self.rnn_dp, self.bidirectional, self.act,
                  self.dp, self.include_lanes)

