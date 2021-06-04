from ForecastModels.ForecastModel import *
import progressbar

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_BS = 256


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


class MLPModel(ForecastModel):
    def __init__(self, act='selu', dp=None, epochs=10, callback=False, l2_reg=False, **kwargs):

        self.dp = dp if dp is not None else 0.0
        self.epochs = epochs
        self.callback = callback
        self.l2_reg = l2_reg

        self.swap_batch_dim = False

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

        self.loss = torch.nn.MSELoss()

        super().__init__(**kwargs)

        print("Built model with {:,} parameters".format(sum(p.numel() for p in self.model.parameters())))

        self.optimizer = Adam(self.model.parameters(), weight_decay=l2_reg)

        self.name = '_'.join([self.name, "%f" % self.dp, act, str(l2_reg)])

    def _gen_model(self):
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
                x = [v.swapaxes(0, 1 if self.swap_batch_dim else 0).to(DEVICE) for v in x]
                y = y.swapaxes(0, 1 if self.swap_batch_dim else 0).to(DEVICE)
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
