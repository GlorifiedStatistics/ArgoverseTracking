from ForecastModels.ForecastModel import *
import progressbar

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_BS = 256


class MLPModel(ForecastModel):
    def __init__(self, act='elu', dp=None, loss='mse', epochs=10, callback=False, l2_reg=False, **kwargs):

        self.dp = dp if dp is not None else 0.0
        self.epochs = epochs
        self.callback = callback
        self.l2_reg = l2_reg

        act = act.lower()
        if act in ['sig', 'sigmoid']:
            self.act = torch.sigmoid
        elif act in ['relu']:
            self.act = torch.relu
        elif act in ['selu']:
            self.act = torch.selu
        elif act in ['tanh']:
            self.act = torch.tanh
        else:
            raise ValueError("Unknown activation: %s" % act)

        loss = loss.lower()
        if loss in ['mse']:
            self.loss = torch.nn.MSELoss()
        else:
            raise ValueError("Unknown loss: %s" % loss)

        super().__init__(**kwargs)

        print("Built model with {:,} parameters".format(sum(p.numel() for p in self.model.parameters())))

        self.optimizer = Adam(self.model.parameters(), weight_decay=l2_reg)

        self.name = '_'.join([self.name, "%f" % self.dp])

    def _gen_model(self):
        class _M(nn.Module):
            def __init__(self, in_size, act, dp=None):
                super().__init__()
                self.ff1 = nn.Linear(in_size, 3000)
                self.ff2 = nn.Linear(3000, 1000)
                self.ff3 = nn.Linear(1000, 300)
                self.ff4 = nn.Linear(300, 60)
                self.bn1 = nn.BatchNorm1d(3000)
                self.bn2 = nn.BatchNorm1d(1000)
                self.bn3 = nn.BatchNorm1d(300)
                self.dp = dp
                self.act = act

            def forward(self, *args):
                x = args[0]

                def _dp(v):
                    return nn.Dropout(self.dp)(v) if self.dp is not None and self.dp > 0 else v
                x = _dp(self.act(self.bn1(self.ff1(x))))
                x = _dp(self.act(self.bn2(self.ff2(x))))
                x = _dp(self.act(self.bn3(self.ff3(x))))
                x = self.act(self.ff4(x))
                return x

        return _M(self.train_x.shape[1], self.act, self.dp)

    def _train_model(self):
        datasets = [Tensor(x) for x in self.train_x] if isinstance(self.train_x, (list, tuple)) else \
            [Tensor(self.train_x)]
        datasets.append(Tensor(self.train_y))

        self.model.to(DEVICE)
        self.model.train()

        for epoch in range(self.epochs):
            total_loss = 0
            print("Traning epoch: %d / %d" % (epoch + 1, self.epochs))
            for i, (*x, y) in enumerate(progressbar.progressbar(DataLoader(TensorDataset(*datasets), batch_size=_BS))):
                x = [v.to(DEVICE) for v in x]
                y = y.to(DEVICE)
                self.optimizer.zero_grad()

                outputs = self.model(*x)
                loss = self.loss(outputs, y)
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            print("Loss: %f\n" % (total_loss / (i + 1)))

            if self.callback:
                self.pred_metrics()

    def _predict(self, X):
        self.model.to(DEVICE)
        self.model.eval()
        inputs = [Tensor(x) for x in X] if isinstance(X, (list, tuple)) else [Tensor(X)]

        pred = None
        with torch.no_grad():
            for x in progressbar.progressbar(DataLoader(TensorDataset(*inputs), batch_size=_BS)):
                x = [v.to(DEVICE) for v in x]
                outputs = self.model(*x)
                outputs = outputs.numpy() if not torch.cuda.is_available() else outputs.cpu().numpy()
                if pred is None:
                    pred = outputs.copy()
                else:
                    pred = np.append(pred, outputs.copy(), axis=0)
        return pred

    def save(self):
        path = self._model_path()
        print("Saving NN model to: '%s'" % path)
        torch.save(self.model, path)

    def load(self):
        path = self._model_path()
        print("Loading NN model from: '%s'" % path)
        return torch.load(path)


class RNNModel(MLPModel):
    def __init__(self, rnn_size=256, num_rnn_layers=1, rnn_dp=None, rnn_input_dp=None, bidirectional=False,
                 attention=False, **kwargs):

        self.rnn_size = rnn_size
        self.rnn_dp = rnn_dp if rnn_dp is not None else 0
        self.rnn_input_dp = rnn_input_dp if rnn_input_dp is not None else 0
        self.bidirectional = bidirectional
        self.attention = attention
        self.num_rnn_layers = num_rnn_layers

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

    def _input_sizes(self):
        x = [19, 60 * (5 if self.extra_features else 4)]
        return [x, [4 * MAX_LANES]] if self.include_lanes else [x]

    def _gen_model(self):
        class _M(nn.Module):
            def __init__(self, input_size, rnn_size, num_rnn_layers, rnn_dp, bi, act, dp, include_lanes):
                super().__init__()
                input_size, *lane_size = input_size
                self.rnn_size = rnn_size
                self.bi = bi
                self.num_rnn_layers = num_rnn_layers
                self.lstm = nn.LSTM(input_size[1], hidden_size=rnn_size, dropout=rnn_dp,
                                    bidirectional=bi, batch_first=True, num_layers=num_rnn_layers)
                rs = self.rnn_size * num_rnn_layers * (2 if bi else 1)
                self.ff1 = nn.Linear(rs + lane_size[0][0], 512) if include_lanes else nn.Linear(rs, 512)
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

                x, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
                x = x[:, -1, :].contiguous().view(-1, self.rnn_size * (2 if self.bi else 1))

                if self.include_lanes:
                    x = torch.cat((x, other[0]), dim=1)

                x = _dp(self.act(self.bn1(self.ff1(x))))
                x = _dp(self.act(self.bn2(self.ff2(x))))
                x = self.act(self.ff3(x))

                return x

        return _M(self._input_sizes(), self.rnn_size, self.num_rnn_layers, self.rnn_dp, self.bidirectional, self.act,
                  self.dp, self.include_lanes)


class ResnetModel(RNNModel):
    def __init__(self, batch_norm_2d=False, **kwargs):
        kwargs['include_lanes'] = True
        kwargs['extra_features'] = True
        self.bn = batch_norm_2d
        super().__init__(**kwargs)

        self.name += '_' + str(self.bn)

    def _conv_data(self, data):
        data = super()._conv_data(data)

        for k in ['train_x', 'test_x', 'val_x']:
            p = data[k][0][:, :, :60 * 2].reshape([-1, 19, 60, 2])
            v = data[k][0][:, :, 60 * 2: 60 * 4].reshape([-1, 19, 60, 2])
            s = data[k][0][:, :, 60 * 4:].reshape([-1, 19, 60, 1])

            data[k] = [p, v, s, data[k][-1]]
            data[k] = [np.swapaxes(x, 1, 3) for x in data[k][:-1]] + [data[k][-1]]

        return data

    def _gen_model(self):
        class _M(nn.Module):
            def __init__(self, act, bn, dp):
                super().__init__()
                self.all_c1 = nn.Conv2d(5, 32, 5, padding=2)
                self.all_bn1 = nn.BatchNorm2d(32) if bn else None
                self.all_c2 = nn.Conv2d(32, 64, 3, padding=1)
                self.all_bn2 = nn.BatchNorm2d(64) if bn else None

                self.all_c3 = nn.Conv2d(69, 64, 5)
                self.all_bn3 = nn.BatchNorm2d(64) if bn else None

                self.all_c4 = nn.Conv2d(64, 64, 3, padding=1)
                self.all_bn4 = nn.BatchNorm2d(64) if bn else None
                self.all_c5 = nn.Conv2d(64, 64, 3, padding=1)
                self.all_bn5 = nn.BatchNorm2d(64) if bn else None

                self.all_c6 = nn.Conv2d(128, 64, 5)
                self.all_bn6 = nn.BatchNorm2d(64) if bn else None

                self.all_c7 = nn.Conv2d(64, 64, 3, padding=1)
                self.all_bn7 = nn.BatchNorm2d(64) if bn else None
                self.all_c8 = nn.Conv2d(64, 64, 3, padding=1)
                self.all_bn8 = nn.BatchNorm2d(64) if bn else None

                self.all_c9 = nn.Conv2d(128, 64, 5)
                self.all_bn9 = nn.BatchNorm2d(64) if bn else None

                self.all_c10 = nn.Conv2d(64, 64, 3, padding=1)
                self.all_bn10 = nn.BatchNorm2d(64) if bn else None
                self.all_c11 = nn.Conv2d(64, 64, 3, padding=1)
                self.all_bn11 = nn.BatchNorm2d(64) if bn else None

                self.all_c12 = nn.Conv2d(128, 64, 7)
                self.all_bn12 = nn.BatchNorm2d(64) if bn else None

                self.all_c13 = nn.Conv2d(64, 64, 3, padding=1)
                self.all_bn13 = nn.BatchNorm2d(64) if bn else None
                self.all_c14 = nn.Conv2d(64, 64, 3, padding=1)
                self.all_bn14 = nn.BatchNorm2d(64) if bn else None

                self.all_view = 64 * 42 * 1
                self.all_fc1 = nn.Linear(self.all_view, 2048)
                self.all_fcbn1 = nn.BatchNorm1d(2048) if bn else None
                self.all_fc2 = nn.Linear(2048, 1024)
                self.all_fcbn2 = nn.BatchNorm1d(1024) if bn else None

                self.fc1 = nn.Linear(1024 + 4 * MAX_LANES, 1024)
                self.bn1 = nn.BatchNorm1d(1024)

                self.fc2 = nn.Linear(1024, 256)
                self.bn2 = nn.BatchNorm1d(256)

                self.fc3 = nn.Linear(256, 128)
                self.bn3 = nn.BatchNorm1d(128)

                self.fc4 = nn.Linear(128, 60)

                self.dp2d = nn.Dropout2d(dp)
                self.dp1d = nn.Dropout(dp)

                self.dp = dp > 0
                self.bn = bn
                self.act = act

            def forward(self, *args):
                def _F2d(z, bnl):
                    if self.bn:
                        z = bnl(z)
                    z = self.act(z)
                    if self.dp:
                        z = self.dp2d(z)
                    return z

                def _F1d(z, bnl, last=False):
                    if self.bn and not last:
                        z = bnl(z)
                    z = self.act(z)
                    if self.dp and not last:
                        z = self.dp1d(z)
                    return z

                p, v, s, lanes = args

                comb = torch.cat((p, v, s), dim=1)

                a = _F2d(self.all_c1(comb), self.all_bn1)
                a = _F2d(self.all_c2(a), self.all_bn2)

                skip1 = _F2d(self.all_c3(torch.cat((a, comb), dim=1)), self.all_bn3)

                a = _F2d(self.all_c4(skip1), self.all_bn4)
                a = _F2d(self.all_c5(a), self.all_bn5)

                skip2 = _F2d(self.all_c6(torch.cat((a, skip1), dim=1)), self.all_bn6)

                a = _F2d(self.all_c7(skip2), self.all_bn7)
                a = _F2d(self.all_c8(a), self.all_bn8)

                skip3 = _F2d(self.all_c9(torch.cat((a, skip2), dim=1)), self.all_bn9)

                a = _F2d(self.all_c10(skip3), self.all_bn10)
                a = _F2d(self.all_c11(a), self.all_bn11)

                skip4 = _F2d(self.all_c12(torch.cat((a, skip3), dim=1)), self.all_bn12)

                a = _F2d(self.all_c13(skip4), self.all_bn13)
                a = _F2d(self.all_c14(a), self.all_bn14).contiguous().view(-1, self.all_view)

                a = _F1d(self.all_fc1(a), self.all_fcbn1)
                a = _F1d(self.all_fc2(a), self.all_fcbn2)

                x = torch.cat((a, lanes), dim=1)
                x = _F1d(self.fc1(x), self.bn1)
                x = _F1d(self.fc2(x), self.bn2)
                x = _F1d(self.fc3(x), self.bn3)
                x = _F1d(self.fc4(x), None, last=True)
                return x

        return _M(self.act, self.dp, self.bn)