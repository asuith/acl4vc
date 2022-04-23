import torch
import torch.nn as nn
from typing import Optional        

__all__ = (
    'Predictor'
)


def get_predictor(opt: dict) -> Optional[nn.Module]:
    skip_crit_names = ['lang', 'attention_weight', 'attribute']

    layers = []
    for crit_name in opt['crit']:
        if crit_name in skip_crit_names:
            continue

        class_name = 'Predictor_{}'.format(crit_name)
        if class_name not in globals():
            raise ValueError('We can not find the class `{}` in {}'.format(class_name, __file__))

        layers.append(globals()[class_name](opt))
    
    if not len(layers):
        return None
    
    return Predictor(layers)


class Predictor(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, encoder_hidden_states, **kwargs):
        results = {}
        for layer in self.layers:
            results.update(layer(encoder_hidden_states, **kwargs))
        return results


class Predictor_length(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.net = nn.Sequential(
                    nn.Linear(opt['dim_hidden'], opt['dim_hidden']),
                    nn.ReLU(),
                    nn.Dropout(opt['hidden_dropout_prob']),
                    nn.Linear(opt['dim_hidden'], opt['max_len']),
                )

    def forward(self, encoder_hidden_states, **kwargs):
        if isinstance(encoder_hidden_states, list):
            assert len(encoder_hidden_states) == 1
            encoder_hidden_states = encoder_hidden_states[0]
        assert len(encoder_hidden_states.shape) == 3

        out = self.net(encoder_hidden_states.mean(1))
        return {'preds_length': torch.log_softmax(out, dim=-1)}
