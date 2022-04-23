from collections import defaultdict
from typing import Dict
import numpy as np
from config import Constants
import torch
import torch.nn as nn
from .bert import BertEmbeddings, BertLayer
from torch.nn import Parameter

__all__ = (
    'BertDecoder', 
    'BertDecoderDisentangled',
    'TransformerDecoder',
    'SingleLayerRNNDecoder',
)


def get_decoder(opt: dict) -> nn.Module:
    class_name = opt['decoder']
    if class_name not in globals():
        raise ValueError('We can not find the class `{}` in {}'.format(class_name, __file__))

    return globals()[class_name](opt)


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


def get_subsequent_mask(seq, watch=0):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    if watch != 0 and len_s >= watch:
        assert watch > 0
        tmp = torch.tril(torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=-watch)
    else:
        tmp = None

    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    if tmp is not None:
        subsequent_mask += tmp
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask


def resampling(source, tgt_tokens):
    pad_mask = tgt_tokens.eq(Constants.PAD)
    length = (1 - pad_mask).sum(-1)
    bsz, seq_len = tgt_tokens.shape

    all_idx = []
    scale = source.size(1) / length.float()
    for i in range(bsz):
        idx = (torch.arange(0, seq_len, device=tgt_tokens.device).float() * scale[i].repeat(seq_len)).long()
        max_idx = tgt_tokens.new(seq_len).fill_(source.size(1) - 1)
        idx = torch.where(idx < source.size(1), idx, max_idx)
        all_idx.append(idx)
    all_idx = torch.stack(all_idx, dim=0).unsqueeze(2).repeat(1, 1, source.size(2))
    return source.gather(1, all_idx)


class EmptyObject(object):
    def __init__(self):
        pass


def dict2obj(dict):
    obj = EmptyObject()
    obj.__dict__.update(dict)
    return obj


class BertDecoder(nn.Module):
    def __init__(self, config, embedding=None):
        super(BertDecoder, self).__init__()
        if isinstance(config, dict):
            config = dict2obj(config)
        self.embedding = BertEmbeddings(config, return_pos=True if config.pos_attention else False) if embedding is None else embedding
        self.layer = nn.ModuleList([BertLayer(config, is_decoder_layer=True) for _ in range(config.num_hidden_layers_decoder)])
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.pos_attention = config.pos_attention
        self.enhance_input = config.enhance_input
        self.watch = config.watch

        self.decoding_type = config.decoding_type

    def _init_embedding(self, weight, option={}, is_numpy=False):
        if is_numpy:
            self.embedding.word_embeddings.weight.data = 0
        else:
            self.embedding.word_embeddings.weight.data.copy_(weight.data)
        if not option.get('train_emb', False):
            for p in self.embedding.word_embeddings.parameters():
                p.requires_grad = False

    def get_word_embeddings(self):
        return self.embedding.word_embeddings.weight

    def set_word_embeddings(self, we):
        raise NotImplementedError('not support right now')

    def forward(self, input_ids, encoder_hidden_states=None, category=None, **kwargs):
        decoding_type = kwargs.get('decoding_type', self.decoding_type)
        output_attentions = kwargs.get('output_attentions', False)

        if isinstance(encoder_hidden_states, list):
            assert len(encoder_hidden_states) == 1
            encoder_hidden_states = encoder_hidden_states[0]
        all_attentions = ()

        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=input_ids, seq_q=input_ids)
        if decoding_type == 'NARFormer':
            slf_attn_mask = slf_attn_mask_keypad
        elif decoding_type == 'SelfMask':
            slf_attn_mask = slf_attn_mask_keypad
            seq_len = input_ids.size(1)
            
            diag =  torch.tril(torch.ones((seq_len, seq_len), device=slf_attn_mask.device, dtype=torch.uint8), diagonal=0) & \
                    torch.triu(torch.ones((seq_len, seq_len), device=slf_attn_mask.device, dtype=torch.uint8), diagonal=0)
            slf_attn_mask = (slf_attn_mask + diag).gt(0)

            # the i-th target can not see itself from the inputs
            '''
            tokens: <bos>   a       girl    is      singing <eos>
            target: a       girl    is      singing <eos>   ..
            '''
            #print(slf_attn_mask[0], slf_attn_mask.shape)
        else:
            slf_attn_mask_subseq = get_subsequent_mask(input_ids, watch=self.watch)
            slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        non_pad_mask = get_non_pad_mask(input_ids)
        src_seq = torch.ones(encoder_hidden_states.size(0), encoder_hidden_states.size(1)).to(encoder_hidden_states.device)
        attend_to_enc_output_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=input_ids)

        additional_feats = None
        if decoding_type == 'NARFormer':
            if self.enhance_input == 0:
                pass
            elif self.enhance_input == 1:
                additional_feats = resampling(encoder_hidden_states, input_ids)
            elif self.enhance_input == 2:
                additional_feats = encoder_hidden_states.mean(1).unsqueeze(1).repeat(1, input_ids.size(1), 1)
            else:
                raise ValueError('enhance_input shoud be either 0, 1 or 2')

        if self.pos_attention:
            hidden_states, position_embeddings = self.embedding(input_ids, category=category)
        else:
            hidden_states = self.embedding(input_ids, additional_feats=additional_feats, category=category)
            position_embeddings = None

        res = []
        for i, layer_module in enumerate(self.layer):
            if not i:
                input_ = hidden_states
            else:
                input_ = layer_outputs[0]# + hidden_states
            
            layer_outputs = layer_module(
                input_, 
                non_pad_mask=non_pad_mask, 
                attention_mask=slf_attn_mask,
                encoder_hidden_states=encoder_hidden_states, 
                attend_to_enc_output_mask=attend_to_enc_output_mask, 
                position_embeddings=position_embeddings, 
                word_embeddings=self.get_word_embeddings(),
                **kwargs
            )

            res.append(layer_outputs[0])
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)
            embs = layer_outputs[1]

        return {
            'hidden_states': self.dropout(res[-1]),
            'all_hidden_states': res,
            'all_attentions': all_attentions,
        }


class BertDecoderDisentangled(nn.Module):
    def __init__(self, config):
        super(BertDecoderDisentangled, self).__init__()
        if isinstance(config, dict):
            config = dict2obj(config)
        self.bert = BertDecoder(config)

    def get_word_embeddings(self):
        return self.bert.get_word_embeddings()

    def set_word_embeddings(self, we):
        self.bert.set_word_embeddings(we)

    def forward(self, input_ids, encoder_hidden_states, category, **kwargs):
        if isinstance(encoder_hidden_states, list):
            assert len(encoder_hidden_states) == 1
            encoder_hidden_states = encoder_hidden_states[0]

        if isinstance(input_ids, list):
            # training
            assert len(input_ids) == 2
            outputs1 = self.bert(input_ids[0], encoder_hidden_states, category, **kwargs)
            outputs2 = self.bert(input_ids[1], encoder_hidden_states, category, **kwargs)
            return {'hidden_states': [outputs1['hidden_states'], outputs2['hidden_states']]}
        else:
            # evaluation
            return self.bert(input_ids, encoder_hidden_states, category, **kwargs)


class RNNDecoderBase(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

    def prepare_word_embeddings(self):
        if self.opt.get('pretrained_embs_path', ''):
            # the path to pretrained word embs is specified
            module = nn.Embedding.from_pretrained(
                embeddings=torch.from_numpy(np.load(self.opt['pretrained_embs_path'])).float(),
                freeze=True,
                padding_idx=Constants.PAD,
            )
            assert module.weight.shape[0] == self.opt['vocab_size']    
        else:
            module = nn.Embedding(self.opt['vocab_size'], self.opt['dim_hidden'], padding_idx=Constants.PAD)
        return module

    def get_word_embeddings(self):
        return self.embedding.weight.data

    def set_word_embeddings(self, data):
        raise NotImplementedError('Not support right now!')

    def init_lstm_forget_bias(self, forget_bias):
        print("====> forget bias %g" % (forget_bias))
        # TODO: make it more elegant
        for module in self.children():
            if hasattr(module, 'bias_ih') and hasattr(module, 'bias_hh'):
                # We do not decide if a module is a LSTM (cell) by its name, but the attributes it has
                # when a module has `bias_ih` and `bias_hh`, it is quite possible a LSTM (cell)
                ingate, forgetgate, cellgate, outgate = module.bias_ih.chunk(4, 0)
                forgetgate = forgetgate + forget_bias / 2
                module.bias_ih = Parameter(torch.cat([ingate, forgetgate, cellgate, outgate], dim=0))

                ingate, forgetgate, cellgate, outgate = module.bias_hh.chunk(4, 0)
                forgetgate = forgetgate + forget_bias / 2
                module.bias_hh = Parameter(torch.cat([ingate, forgetgate, cellgate, outgate], dim=0))
        
    def init_decoder_rnn_hidden_states(self, encoder_hidden_states):
        assert hasattr(self, 'rnn_type'), 'Please make sure the derived classes have `self.rnn_type` (str, `lstm` or `gru`)'
        assert hasattr(self, 'v2h'), 'Please make sure the derived classes have `self.v2h` (None or nn.Module)'

        mean_v = self.get_mean_video_features(encoder_hidden_states)

        if self.rnn_type == 'lstm':
            assert hasattr(self, 'v2c'), 'Please make sure the derived classes have `self.v2h` (None or nn.Module)'
            hidden = mean_v if self.v2h is None else self.v2h(mean_v)
            cell = mean_v if self.v2c is None else self.v2c(mean_v)
            decoder_rnn_hidden_states = (hidden, cell)
        else:
            decoder_rnn_hidden_states = mean_v if self.v2h is None else self.v2h(mean_v)

        if hasattr(self, 'init_decoder_rnn_hidden_states_post_processing'):
            return self.init_decoder_rnn_hidden_states_post_processing(decoder_rnn_hidden_states)
        
        return decoder_rnn_hidden_states

    def preparation_before_feedforward(self, decoder_rnn_hidden_states, encoder_hidden_states):
        if decoder_rnn_hidden_states is None:
            decoder_rnn_hidden_states = self.init_decoder_rnn_hidden_states(encoder_hidden_states)

        return decoder_rnn_hidden_states, encoder_hidden_states

    def get_mean_video_features(self, encoder_hidden_states):
        if not isinstance(encoder_hidden_states, list):
            encoder_hidden_states = [encoder_hidden_states]

        mean_v = torch.stack(encoder_hidden_states, dim=0).mean(0)
        mean_v = mean_v.mean(1) # [bsz, dim_hidden]
        return mean_v

    def get_hidden_states(self, decoder_rnn_hidden_states):
        # the function `init_decoder_rnn_hidden_states` has confirmed hasattr(self, 'rnn_type')
        if self.rnn_type == 'lstm':
            hidden_states = decoder_rnn_hidden_states[0]
        else:
            hidden_states = decoder_rnn_hidden_states
        
        if len(hidden_states.shape) == 3:
            assert hidden_states.size(0) == 1
            hidden_states = hidden_states.squeeze(0)

        return hidden_states

    def forward_step(self, it, encoder_hidden_states, decoder_rnn_hidden_states=None, category=None, **kwargs):
        raise NotImplementedError('Please implement `forward_step` in the derived classes')
    
    def forward(self, input_ids, encoder_hidden_states, category=None, **kwargs):
        # teacher forcing loop, i.e., without schedule sampling
        # schedule sampling is implemented in the class `RNNSeq2Seq` in models/Framework.py
        assert input_ids.dim() == 2, "(bsz, seq_len)"

        hidden_states = []
        attention_probs = []
        other_component_outputs = defaultdict(list)

        decoder_rnn_hidden_states = None
        for i in range(input_ids.size(1)):
            it = input_ids[:, i]
            decoding_phase_outputs = self.forward_step(it, encoder_hidden_states, decoder_rnn_hidden_states, category, **kwargs)

            decoder_rnn_hidden_states = decoding_phase_outputs.pop('decoder_rnn_hidden_states')
            hidden_states.append(decoding_phase_outputs.pop('hidden_states'))
            attention_probs.append(decoding_phase_outputs.pop('attention_probs'))

            for k, v in decoding_phase_outputs.items():
                other_component_outputs[k].append(v)

        for k in other_component_outputs:
            other_component_outputs[k] = torch.stack(other_component_outputs[k], dim=1)
        
        return {
            'hidden_states': torch.stack(hidden_states, dim=1), # [bsz, max_len-1, dim_hidden]
            'attention_probs': torch.stack(attention_probs, dim=2), # [bsz, num_feats, max_len-1, n_frames]
            **other_component_outputs
        }



from models.components.Attention import AdditiveAttention, MultiLevelAttention
class SingleLayerRNNDecoder(RNNDecoderBase):
    def __init__(self, opt):
        super().__init__(opt)
        # prelimenary
        num_modality = len(opt['modality']) if opt['fusion'] != 'temporal_concat' else 1

        self.with_category = opt['with_category']
        num_category = opt.get('num_category', 0)
        if self.with_category:
            assert num_category != 0
        else:
            num_category = 0

        # define the word embeddings
        self.embedding = self.prepare_word_embeddings()
        dim_word = self.embedding.weight.shape[1]

        # define the rnn module
        self.rnn_type = opt.get('rnn_type', 'lstm').lower()
        rnn_func = nn.LSTMCell if self.rnn_type == 'lstm' else nn.GRUCell

        self.rnn = rnn_func(
            # inputs: [y(t-1); att(feats); category]
            input_size=dim_word + opt['dim_hidden'] * num_modality + num_category,
            hidden_size=opt['dim_hidden']
        )

        self.v2h = nn.Linear(opt['dim_hidden'], opt['dim_hidden']) # to init h0
        if self.rnn_type == 'lstm':
            self.v2c = nn.Linear(opt['dim_hidden'], opt['dim_hidden']) # to init c0
            self.init_lstm_forget_bias(opt.get('forget_bias', 0.6))

        # define the attention module
        att_func = MultiLevelAttention if opt.get('with_multileval_attention', False) \
            else AdditiveAttention

        self.att = att_func(
            dim_hidden=opt['dim_hidden'],
            dim_feats=[opt['dim_hidden']] * num_modality,
            dim_mid=opt['dim_hidden'],
            feats_share_weights=opt.get('feats_share_weights', False)
        )
        self.dropout = nn.Dropout(opt['hidden_dropout_prob'])

    def forward_step(self, it, encoder_hidden_states, decoder_rnn_hidden_states=None, category=None, **kwargs):
        assert it.dim() == 1, '(bsz, )'

        decoder_rnn_hidden_states, encoder_hidden_states = self.preparation_before_feedforward(
            decoder_rnn_hidden_states, encoder_hidden_states)

        # feed forward
        context, attention_probs = self.att(
            hidden_states=self.get_hidden_states(decoder_rnn_hidden_states), # use h(t-1) as the query
            feats=encoder_hidden_states,
            **kwargs,
        )

        rnn_inputs = [self.embedding(it), context] + ([category] if self.with_category else [])
        rnn_inputs = self.dropout(torch.cat(rnn_inputs, dim=-1))
        decoder_rnn_hidden_states = self.rnn(rnn_inputs, decoder_rnn_hidden_states)

        final_hidden_states = self.get_hidden_states(decoder_rnn_hidden_states) # get h(t)

        main_outputs = {
            'hidden_states': self.dropout(final_hidden_states),
            'decoder_rnn_hidden_states': decoder_rnn_hidden_states,
            'attention_probs': attention_probs
        }

        other_components_outputs = self.run_other_components(
            context=context, 
            input_ids=it, 
            hidden_states=final_hidden_states
        )
        return {**main_outputs, **other_components_outputs}


class TwoLayerRNNDecoderBase(RNNDecoderBase):
    def forward_step(self, it, encoder_hidden_states, decoder_rnn_hidden_states=None, category=None, **kwargs):
        assert it.dim() == 1, '(bsz, )'
        assert hasattr(self, 'bottom_rnn')
        assert hasattr(self, 'top_rnn')

        decoder_rnn_hidden_states, encoder_hidden_states = self.preparation_before_feedforward(
            decoder_rnn_hidden_states, encoder_hidden_states)

        bottom_rnn_inputs, other_outputs_bottom = self.prepare_bottom_rnn_inputs(
            it, decoder_rnn_hidden_states, encoder_hidden_states, category, **kwargs)
        decoder_rnn_hidden_states[0] = self.bottom_rnn(bottom_rnn_inputs, decoder_rnn_hidden_states[0])

        top_rnn_inputs, other_outputs_top = self.prepare_top_rnn_inputs(
            decoder_rnn_hidden_states, encoder_hidden_states, **kwargs)
        decoder_rnn_hidden_states[1] = self.top_rnn(top_rnn_inputs, decoder_rnn_hidden_states[1])

        final_hidden_states, other_outputs_final = self.prepare_final_hidden_states(
            decoder_rnn_hidden_states, **other_outputs_bottom, **other_outputs_top)

        return {
            'hidden_states': self.dropout(final_hidden_states),
            'decoder_rnn_hidden_states': decoder_rnn_hidden_states,
            **other_outputs_bottom,
            **other_outputs_top,
            **other_outputs_final
        }

    def prepare_bottom_rnn_inputs(self, it, decoder_rnn_hidden_states, encoder_hidden_states, category, **kwargs):
        raise NotImplementedError('Please implement the `prepare_bottom_rnn_inputs` function in derived classes')

    def prepare_top_rnn_inputs(self, decoder_rnn_hidden_states, encoder_hidden_states, **kwargs):
        raise NotImplementedError('Please implement the `prepare_top_rnn_inputs` function in derived classes')

    def prepare_final_hidden_states(self, decoder_rnn_hidden_states, **kwargs):
        raise NotImplementedError('Please implement the `prepare_final_hidden_states` function in derived classes')


class TopDownAttentionRNNDecoder(TwoLayerRNNDecoderBase):
    ''' Reproduce the decoder of `Bottom-Up and Top-Down Attention
        for Image Captioning and Visual Question Answering` (CVPR 2018)
        https://arxiv.org/pdf/1707.07998.pdf
    '''
    def __init__(self, opt):
        super().__init__(opt)
        # prelimenary
        num_modality = len(opt['modality']) if opt['fusion'] != 'temporal_concat' else 1

        self.with_category = opt['with_category']
        num_category = opt.get('num_category', 0)
        if self.with_category:
            assert num_category != 0
        else:
            num_category = 0

        # define the word embeddings
        self.embedding = self.prepare_word_embeddings()
        dim_word = self.embedding.weight.shape[1]

        # define the rnn module
        self.rnn_type = opt.get('rnn_type', 'lstm').lower()
        rnn_func = nn.LSTMCell if self.rnn_type == 'lstm' else nn.GRUCell

        self.bottom_rnn = rnn_func(
            # inputs: [y(t-1); top_h(t-1); mean_v; category]
            input_size=dim_word + opt['dim_hidden'] * 2 + num_category,
            hidden_size=opt['dim_hidden']
        )
        self.top_rnn = rnn_func(
            # inputs: [bottom_h(t); att(feats)]
            input_size=opt['dim_hidden'] + opt['dim_hidden'] * num_modality,
            hidden_size=opt['dim_hidden']
        )

        if self.rnn_type == 'lstm':
            self.init_lstm_forget_bias(opt.get('forget_bias', 0.6))

        # define the attention module
        att_func = MultiLevelAttention if opt.get('with_multileval_attention', False) \
            else AdditiveAttention

        self.att = att_func(
            dim_hidden=opt['dim_hidden'],
            dim_feats=[opt['dim_hidden']] * num_modality,
            dim_mid=opt['dim_hidden'],
            feats_share_weights=opt.get('feats_share_weights', False)
        )
        self.dropout = nn.Dropout(opt['hidden_dropout_prob'])

    def init_decoder_rnn_hidden_states(self, encoder_hidden_states):
        if isinstance(encoder_hidden_states, list):
            bsz = encoder_hidden_states[0].size(0)
        else:
            bsz = encoder_hidden_states.size(0)

        weight = next(self.parameters())
        if self.rnn_type == 'lstm':
            return [
                (weight.new_zeros([bsz, self.bottom_rnn.hidden_size]), weight.new_zeros([bsz, self.bottom_rnn.hidden_size])),
                (weight.new_zeros([bsz, self.top_rnn.hidden_size]), weight.new_zeros([bsz, self.top_rnn.hidden_size]))
            ]
        return [weight.new_zeros([bsz, self.bottom_rnn.hidden_size]), weight.new_zeros([bsz, self.top_rnn.hidden_size])]

    def prepare_bottom_rnn_inputs(self, it, decoder_rnn_hidden_states, encoder_hidden_states, category, **kwargs):
        inputs = [
            self.embedding(it),
            self.get_hidden_states(decoder_rnn_hidden_states[1]), # top_h(t-1)
            self.get_mean_video_features(encoder_hidden_states),
        ] + ([category] if self.with_category else [])

        inputs = self.dropout(torch.cat(inputs, dim=-1))
        return inputs, {}

    def prepare_top_rnn_inputs(self, decoder_rnn_hidden_states, encoder_hidden_states, **kwargs):
        attention_rnn_hidden_states = self.get_hidden_states(decoder_rnn_hidden_states[0]) # bottom_h(t)

        context, attention_probs = self.att(
            hidden_states=attention_rnn_hidden_states, # bottom_h(t) as the query
            feats=encoder_hidden_states,
            **kwargs
        )

        inputs = self.dropout(torch.cat([attention_rnn_hidden_states, context], dim=-1))
        return inputs, {'attention_probs': attention_probs}

    def prepare_final_hidden_states(self, decoder_rnn_hidden_states, **kwargs):
        return self.get_hidden_states(decoder_rnn_hidden_states[1]), {}


class AdjustedAttentionRNNDecoder(TwoLayerRNNDecoderBase):
    ''' Reproduce the decoder of `Hierarchical LSTM with
        Adjusted Temporal Attention for Video Captioning` (IJCAI 2017)
        https://www.ijcai.org/proceedings/2017/381
    '''
    def __init__(self, opt):
        super().__init__(opt)
        assert opt['fusion'] == 'temporal_concat'

        self.with_category = opt['with_category']
        num_category = opt.get('num_category', 0)
        if self.with_category:
            assert num_category != 0
        else:
            num_category = 0

        # define the word embeddings
        self.embedding = self.prepare_word_embeddings()
        dim_word = self.embedding.weight.shape[1]

        # define the rnn module
        self.rnn_type = opt.get('rnn_type', 'lstm').lower()
        rnn_func = nn.LSTMCell if self.rnn_type == 'lstm' else nn.GRUCell

        self.bottom_rnn = rnn_func(
            # inputs: [y(t-1); category]
            input_size=dim_word + num_category,
            hidden_size=opt['dim_hidden']
        )
        self.top_rnn = rnn_func(
            # inputs: [bottom_h(t)]
            input_size=opt['dim_hidden'],
            hidden_size=opt['dim_hidden']
        )
        self.v2h = nn.Linear(opt['dim_hidden'], opt['dim_hidden']) # init bottom_h(0)
        self.v2c = nn.Linear(opt['dim_hidden'], opt['dim_hidden']) # init bottom_c(0)
        self.gate_layer = nn.Sequential(nn.Linear(opt['dim_hidden'], 1, bias=False), nn.Sigmoid())

        if self.rnn_type == 'lstm':
            self.init_lstm_forget_bias(opt.get('forget_bias', 0.6))

        # define the attention module
        self.att = AdditiveAttention(
            dim_hidden=opt['dim_hidden'],
            dim_feats=opt['dim_hidden'],
            dim_mid=opt['dim_hidden'],
        )
        self.dropout = nn.Dropout(opt['hidden_dropout_prob'])

    def init_decoder_rnn_hidden_states_post_processing(self, decoder_rnn_hidden_states):
        return [decoder_rnn_hidden_states, None]

    def prepare_bottom_rnn_inputs(self, it, decoder_rnn_hidden_states, encoder_hidden_states, category, **kwargs):
        inputs = [self.embedding(it)] + ([category] if self.with_category else [])
        inputs = self.dropout(torch.cat(inputs, dim=-1))
        return inputs, {}

    def prepare_top_rnn_inputs(self, decoder_rnn_hidden_states, encoder_hidden_states, **kwargs):
        attention_rnn_hidden_states = self.get_hidden_states(decoder_rnn_hidden_states[0]) # bottom_h(t)


        context, attention_probs = self.att(
            hidden_states=attention_rnn_hidden_states, # bottom_h(t) as the query
            feats=encoder_hidden_states,
            **kwargs
        )

        inputs = self.dropout(attention_rnn_hidden_states)
        return inputs, {'context': context, 'attention_probs': attention_probs}

    def prepare_final_hidden_states(self, decoder_rnn_hidden_states, context, **kwargs):
        bottom_h = self.get_hidden_states(decoder_rnn_hidden_states[0])
        top_h = self.get_hidden_states(decoder_rnn_hidden_states[1])

        gate = self.gate_layer(bottom_h)
        new_context = gate * context + (1 - gate) * top_h
        final_hidden_states = torch.cat([bottom_h, new_context], dim=-1)

        return final_hidden_states, {'gate': gate}


class ParAAdjustedAttentionRNNDecoder(AdjustedAttentionRNNDecoder):
    ''' Reproduce the decoder of `Hierarchical LSTMs with Adaptive Attention
        for Visual Captioning` (TPAMI 2020)
        https://ieeexplore.ieee.org/document/8620348
        There are 3 arch in the paper (see Fig.4.), and this is the `ParA`.
        For `ConF`, see AdjustedAttentionRNNDecoder.
        `Two-Stream` requires two pretrained AdjustedAttentionRNNDecoder.
    '''
    def __init__(self, opt):
        super().__init__(opt)

        # define the attention module
        self.att_a = AdditiveAttention(
            dim_hidden=opt['dim_hidden'],
            dim_feats=opt['dim_hidden'],
            dim_mid=opt['dim_hidden'],
        )
        self.att_m = AdditiveAttention(
            dim_hidden=opt['dim_hidden'],
            dim_feats=opt['dim_hidden'],
            dim_mid=opt['dim_hidden'],
        )
        del self.att

        self.gate_layer = nn.Sequential(nn.Linear(opt['dim_hidden'], 3, bias=False), nn.Softmax())

    def prepare_top_rnn_inputs(self, decoder_rnn_hidden_states, encoder_hidden_states, **kwargs):
        attention_rnn_hidden_states = self.get_hidden_states(decoder_rnn_hidden_states[0])  # bottom_h(t)

        feature_a, feature_m = torch.chunk(encoder_hidden_states, 2, dim=1)

        context_1, attention_probs_1 = self.att_a(
            hidden_states=attention_rnn_hidden_states, # bottom_h(t) as the query
            feats=feature_a,
            **kwargs
        )

        context_2, attention_probs_2 = self.att_m(
            hidden_states=attention_rnn_hidden_states, # bottom_h(t) as the query
            feats=feature_m,
            **kwargs
        )
        context = torch.stack([context_1, context_2], dim=-1)  # easy to split
        attention_probs = torch.stack([attention_probs_1, attention_probs_2], dim=3)  # used in loss
        inputs = self.dropout(attention_rnn_hidden_states)
        return inputs, {'context': context, 'attention_probs': attention_probs}

    def prepare_final_hidden_states(self, decoder_rnn_hidden_states, context, **kwargs):
        bottom_h = self.get_hidden_states(decoder_rnn_hidden_states[0])
        top_h = self.get_hidden_states(decoder_rnn_hidden_states[1])

        context_1, context_2 = torch.split(context, 1, dim=-1)
        context_1, context_2 = context_1.squeeze(dim=-1), context_2.squeeze(dim=-1)

        beta1, beta2, beta3 = torch.split(self.gate_layer(bottom_h), 1, dim=-1)
        new_context = beta1 * context_1 + beta2 * context_2 + beta3 * top_h
        final_hidden_states = torch.cat([bottom_h, new_context], dim=-1)

        return final_hidden_states, {'gate': [beta1, beta2, beta3]}



class TransformerDecoder(nn.Module):
    def __init__(self, opt):
        super(TransformerDecoder, self).__init__()
        from models.components.Embeddings import Embeddings
        from models.components.Layers import DecoderLayer

        self.embedding = Embeddings(opt)
        self.layers = nn.ModuleList([DecoderLayer(opt) for _ in range(opt['num_hidden_layers_decoder'])])
        self.dropout = nn.Dropout(opt['hidden_dropout_prob'])
        
        self.enhance_input = opt['enhance_input']
        self.decoding_type = opt['decoding_type']

    def _init_embedding(self, weight, option={}, is_numpy=False):
        if is_numpy:
            self.embedding.word_embeddings.weight.data = 0
        else:
            self.embedding.word_embeddings.weight.data.copy_(weight.data)
        if not option.get('train_emb', False):
            for p in self.embedding.word_embeddings.parameters():
                p.requires_grad = False

    def get_word_embeddings(self):
        return self.embedding.word_embeddings.weight

    def set_word_embeddings(self, we):
        raise NotImplementedError('not support right now!')

    def forward(self, input_ids, encoder_hidden_states=None, category=None, head_mask=None, **kwargs):
        decoding_type = kwargs.pop('decoding_type', self.decoding_type)

        if isinstance(encoder_hidden_states, list):
            assert len(encoder_hidden_states) == 1
            encoder_hidden_states = encoder_hidden_states[0]
        
        # get intra-attention (self-attention) mask
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=input_ids, seq_q=input_ids)
        if decoding_type == 'NARFormer':
            attention_mask = slf_attn_mask_keypad
        else:
            slf_attn_mask_subseq = get_subsequent_mask(input_ids)
            attention_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        # get inter-attention (cross-modal attention) mask
        src_seq = torch.ones(encoder_hidden_states.size(0), encoder_hidden_states.size(1)).to(encoder_hidden_states.device)
        encoder_attention_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=input_ids)

        additional_feats = None
        if decoding_type == 'NARFormer':
            if self.enhance_input == 0:
                pass
            elif self.enhance_input == 1:
                additional_feats = resampling(encoder_hidden_states, input_ids)
            elif self.enhance_input == 2:
                additional_feats = encoder_hidden_states.mean(1).unsqueeze(1).repeat(1, input_ids.size(1), 1)
            else:
                raise ValueError('enhance_input shoud be either 0, 1 or 2')

        hidden_states = self.embedding(input_ids, additional_feats=additional_feats, category=category)

        all_hidden_states = [hidden_states]
        all_intra_attentions = ()
        all_inter_attentions = ()

        for layer in self.layers:
            hidden_states, (intra_attention_probs, inter_attention_probs) = layer(
                hidden_states=all_hidden_states[-1], 
                encoder_hidden_states=encoder_hidden_states, 
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask, 
                head_mask=head_mask,
                **kwargs
            )

            all_hidden_states.append(hidden_states)
            all_intra_attentions = all_intra_attentions + (intra_attention_probs, )
            all_inter_attentions = all_inter_attentions + (inter_attention_probs, )

        return {
            'hidden_states': self.dropout(all_hidden_states[-1]),
            'all_hidden_states': all_hidden_states,
            'all_intra_attentions': all_intra_attentions,
            'all_inter_attentions': all_inter_attentions,
        }

