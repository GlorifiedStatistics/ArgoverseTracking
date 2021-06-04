from ForecastModels.Models import *

model = MLPModel(s=False, v=True, lanes=True, norm_func='noop', act='relu', epochs=10, bidirectional=False,
                      num_rnn_layers=1, rnn_dp=None, y_norm_func=None, vmin=-1, vmax=1)
model.train()



