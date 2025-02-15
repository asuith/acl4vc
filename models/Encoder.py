import torch
import torch.nn as nn
from models.components.activations import get_activation


__all__ = (
    'Encoder_HighWay',
    'Encoder_HighWayBN',
    'Encoder_Naive',
    'Encoder_NaiveBN',
)


def get_encoder(opt: dict) -> nn.Module:
    class_name = opt['encoder']
    if class_name not in globals():
        raise ValueError('We can not find the class `{}` in {}'.format(class_name, __file__))

    return globals()[class_name](opt)


class HighWay(nn.Module):
    def __init__(self, hidden_size, with_gate=True):
        super().__init__()
        self.with_gate = with_gate
        self.w1 = nn.Linear(hidden_size, hidden_size)
        if self.with_gate:
            self.w2 = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        #self._init_weights()

    def forward(self, x):
        y = self.tanh(self.w1(x))
        if self.with_gate:
            gate = torch.sigmoid(self.w2(x))
            return gate * x + (1 - gate) * y
        else:
            return x + y


class BN1d(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.bn = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        if len(x.shape) == 2:
            return self.bn(x)
        else:
            assert x.shape[-1] == self.hidden_size
            rest_shape = x.shape[:-1]
            return self.bn(x.contiguous().view(-1, self.hidden_size)).view(*rest_shape, self.hidden_size)


class MultipleStreams(nn.Module):
    def __init__(self, opt, module_func, is_rnn=False, check_valid=False):
        super().__init__()
        self.encoders = []
        modality = opt['modality'].lower()
        for char in modality:
            input_dim = opt.get('dim_' + char, None)
            output_dim = opt.get('dim_hidden', 512)
            dropout = opt.get('encoder_dropout', 0.5)
            assert input_dim is not None, \
                'The modality is {}, but dim_{} can not be found in opt'.format(modality, char)
            
            module = module_func(input_dim, output_dim, dropout, opt)
            self.add_module("Encoder_%s" % char.upper(), module)
            self.encoders.append(module)
 
        self.num_feats = len(modality)
        self.is_rnn = is_rnn

        self.fusion_type = opt.get('fusion_type', 'temporal_concat')
        if check_valid and self.fusion_type not in ['temporal_concat', 'addition', 'none']:
            raise ValueError('We now only support the fusion_type: temporal_concat | addition | none')
        
        if opt.get('visual_memory_early_fusion', False):
            func = MemoryContainer if opt.get('visual_memory_only_predict', False) \
                else MemoryContainerInEncoder
            self.visual_memory = func(opt)

    def forward(self, input_feats: torch.Tensor, **kwargs) -> dict:
        assert self.num_feats == len(input_feats)

        if hasattr(self, 'pre_processing'):
            input_feats = self.pre_processing(input_feats)

        if not self.is_rnn:
            encoder_hidden_states = [encocder(feats) for encocder, feats in zip(self.encoders, input_feats)]
        else:
            # TODO
            pass
        
        results = {'encoder_hidden_states': encoder_hidden_states}

        if hasattr(self, 'post_processing'):
            results.update(self.post_processing(results['encoder_hidden_states']))
        
        if hasattr(self, 'visual_memory'):
            results.update(
                self.visual_memory(
                    word_embeddings=kwargs.get('word_embeddings', None), 
                    **results
                )
            )

        return results
    
    def post_processing(self, encoder_hidden_states: torch.Tensor) -> dict:
        if self.fusion_type != 'none':
            if not isinstance(encoder_hidden_states, list):
                encoder_hidden_states = [encoder_hidden_states]
            if self.fusion_type == 'addition':
                encoder_hidden_states = torch.stack(encoder_hidden_states, dim=0).mean(0)
            elif self.fusion_type == 'temporal_concat':
                encoder_hidden_states = torch.cat(encoder_hidden_states, dim=1)
        
        return {'encoder_hidden_states': encoder_hidden_states}


class Encoder_HighWay(MultipleStreams):
    def __init__(self, opt):
        module_func = lambda x,y,z,opt: nn.Sequential(nn.Linear(x, y), HighWay(y), nn.Dropout(z))
        super().__init__(opt, module_func, check_valid=True)


class Encoder_HighWayBN(MultipleStreams):
    def __init__(self, opt):
        module_func = lambda x,y,z,opt: nn.Sequential(nn.Linear(x, y), HighWay(y), BN1d(y), nn.Dropout(z))
        super().__init__(opt, module_func, check_valid=True)


class Encoder_Naive(MultipleStreams):
    def __init__(self, opt):
        module_func = lambda x,y,z,opt: nn.Sequential(nn.Linear(x, y), nn.ReLU(), nn.Dropout(z))
        super().__init__(opt, module_func, check_valid=True)

class Encoder_Naive_noact(MultipleStreams):
    def __init__(self, opt):
        module_func = lambda x,y,z,opt: nn.Sequential(nn.Linear(x, y), nn.Dropout(z))
        super().__init__(opt, module_func, check_valid=True)

class Encoder_NaiveBN(MultipleStreams):
    def __init__(self, opt):
        module_func = lambda x,y,z,opt: nn.Sequential(nn.Linear(x, y), nn.ReLU(), BN1d(y),  nn.Dropout(z))
        super().__init__(opt, module_func, check_valid=True)


class Encoder_NaiveLN(MultipleStreams):
    def __init__(self, opt):
        module_func = lambda x,y,z,opt: nn.Sequential(nn.Linear(x, y), nn.ReLU(), nn.LayerNorm(y),  nn.Dropout(z))
        super().__init__(opt, module_func, check_valid=True)


class Encoder_NaiveBN_noact(MultipleStreams):
    def __init__(self, opt):
        module_func = lambda x,y,z,opt: nn.Sequential(nn.Linear(x, y), BN1d(y),  nn.Dropout(z))
        super().__init__(opt, module_func, check_valid=True)


class Encoder_NaiveLN_noact(MultipleStreams):
    def __init__(self, opt):
        module_func = lambda x,y,z,opt: nn.Sequential(nn.Linear(x, y), nn.LayerNorm(y),  nn.Dropout(z))
        super().__init__(opt, module_func, check_valid=True)


class Encoder_preBN(MultipleStreams):
    def __init__(self, opt):
        module_func = lambda x,y,z,opt: nn.Sequential(nn.Linear(x, y), nn.ReLU(), BN1d(y),  nn.Dropout(z))
        super().__init__(opt, module_func, check_valid=True)

class Encoder_postBN(MultipleStreams):
    def __init__(self, opt):
        module_func = lambda x,y,z,opt: nn.Sequential(nn.Linear(x, y), nn.ReLU(),  nn.Dropout(z), BN1d(y))
        super().__init__(opt, module_func, check_valid=True)

class Encoder_preLN(MultipleStreams):
    def __init__(self, opt):
        module_func = lambda x,y,z,opt: nn.Sequential(nn.Linear(x, y), nn.ReLU(),  nn.LayerNorm(y), nn.Dropout(z))
        super().__init__(opt, module_func, check_valid=True)

class Encoder_postLN(MultipleStreams):
    def __init__(self, opt):
        module_func = lambda x,y,z,opt: nn.Sequential(nn.Linear(x, y), nn.ReLU(),  nn.Dropout(z), nn.LayerNorm(y))
        super().__init__(opt, module_func, check_valid=True)

class Encoder_preLN_plain(MultipleStreams):
    def __init__(self, opt):
        module_func = lambda x,y,z,opt: nn.Sequential(nn.Linear(x, y),  nn.LayerNorm(y), nn.Dropout(z))
        super().__init__(opt, module_func, check_valid=True)

class Encoder_postLN_plain(MultipleStreams):
    def __init__(self, opt):
        module_func = lambda x,y,z,opt: nn.Sequential(nn.Linear(x, y),  nn.Dropout(z), nn.LayerNorm(y))
        super().__init__(opt, module_func, check_valid=True)


class Encoder_basic(MultipleStreams):
    def __init__(self, opt):
        module_func = lambda x,y,z,opt: nn.Sequential(nn.Linear(x, y), get_activation(opt['encoder_act']), nn.Dropout(z))
        super().__init__(opt, module_func, check_valid=True)


class Encoder_basicLN(MultipleStreams):
    def __init__(self, opt):
        module_func = lambda x,y,z,opt: nn.Sequential(nn.Linear(x, y), get_activation(opt['encoder_act']),  nn.LayerNorm(y), nn.Dropout(z))
        super().__init__(opt, module_func, check_valid=True)


class Encoder_basicBN(MultipleStreams):
    def __init__(self, opt):
        module_func = lambda x,y,z,opt: nn.Sequential(nn.Linear(x, y), get_activation(opt['encoder_act']),  BN1d(y), nn.Dropout(z))
        super().__init__(opt, module_func, check_valid=True)


class TransformerEncoderBase(nn.Module):
    def __init__(self, opt):
        super().__init__()
        from models.components.Embeddings import PositionalEmbedding
        from models.components.Layers import EncoderLayer

        self.trainable_pe = opt.get('trainable_pe', False)
        if self.trainable_pe:
            self.position_embeddings = nn.Embedding(opt['n_frames'], opt['dim_hidden'])
        else:
            self.position_embeddings = PositionalEmbedding(opt['n_frames'], opt['dim_hidden'])
        
        self.LayerNorm = nn.LayerNorm(opt['dim_hidden'], eps=opt['layer_norm_eps'])
        self.dropout = nn.Dropout(opt['hidden_dropout_prob'])

        self.layers = nn.ModuleList([EncoderLayer(opt) for _ in range(opt['num_hidden_layers_encoder'])])
    
    def forward(self, input_feats, only_return_encoder_hidden_states=True):
        if not isinstance(input_feats, list):
            input_feats = [input_feats]

        if self.trainable_pe:
            seq_length = input_feats[0].size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_feats[0].device)
            position_ids = position_ids.unsqueeze(0).repeat(input_feats[0].size(0), 1)
            position_embeddings = self.position_embeddings(position_ids)
        else:
            position_embeddings = self.position_embeddings(input_feats[0])

        hidden_states = [feats + position_embeddings for feats in input_feats]
        hidden_states = torch.cat(hidden_states, dim=1)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        all_encoder_hidden_states = [hidden_states]
        all_encoder_intra_attentions = ()

        for layer in self.layers:
            hidden_states, intra_attention_probs = layer(
                hidden_states=all_encoder_hidden_states[-1], 
                attention_mask=None,
                head_mask=None,
            )

            all_encoder_hidden_states.append(hidden_states)
            all_encoder_intra_attentions = all_encoder_intra_attentions + (intra_attention_probs, )

        if only_return_encoder_hidden_states:
            return all_encoder_hidden_states[-1]
            
        return {
            'encoder_hidden_states': all_encoder_hidden_states[-1],
            'all_encoder_hidden_states': all_encoder_hidden_states,
            'all_encoder_intra_attentions': all_encoder_intra_attentions,
        }


class MultiTransformerEncoder(MultipleStreams):
    def __init__(self, opt):
        module_func = lambda x,y,z,opt: nn.Sequential(nn.Linear(x, y), TransformerEncoderBase(opt))
        super().__init__(opt, module_func, check_valid=True)


class TransformerEncoder(MultipleStreams):
    def __init__(self, opt):
        module_func = lambda x,y,z,opt: nn.Linear(x, y)
        super().__init__(opt, module_func, check_valid=True)
        self.backbone = TransformerEncoderBase(opt)
    
    def post_processing(self, encoder_hidden_states: torch.Tensor) -> dict:
        return self.backbone(encoder_hidden_states, only_return_encoder_hidden_states=False)

    
class Encoder_basicLN(MultipleStreams):
    def __init__(self, opt):
        module_func = lambda x,y,z: nn.Sequential(nn.Linear(x, y), get_activation(opt['encoder_act']),  nn.LayerNorm(y), nn.Dropout(z))
        super().__init__(opt, module_func, check_valid=True)


class Encoder_basicBN(MultipleStreams):
    def __init__(self, opt):
        module_func = lambda x,y,z: nn.Sequential(nn.Linear(x, y), get_activation(opt['encoder_act']),  BN1d(y), nn.Dropout(z))
        super().__init__(opt, module_func, check_valid=True)

