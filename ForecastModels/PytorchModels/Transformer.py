from ForecastModels.PytorchModels.BasicRNN import *


class TransformerModel(BasicRNNModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.swap_batch_dim = True

    def _gen_model(self):
        return nn.Transformer(self.non_lane_shape[-1])

