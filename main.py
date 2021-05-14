from ForecastModels.Models import *


model = TransformerModel(**{'epochs': 10, 'data_type': 'step', 'norm_func': 'tanh', 'batch_norm': True,
        'dp': 0.2, 'act': 'tanh', 'include_lanes': False, 'extra_features': False, 'max_data': None,
                        'attention':True, 'hidden_layers': 2, 'bidirectional': True, 'callback': True})

model.train()
model.compute_val()
