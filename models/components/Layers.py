''' Define the layers in Transformer'''

import torch
import torch.nn as nn
from models.components.SubLayers import (
    MultiHeadAttention, 
    PositionwiseFeedForward
)
from typing import Dict, Any, Optional, Tuple


class EncoderLayer(nn.Module):
    def __init__(self, opt: Dict[str, Any]):
        super(EncoderLayer, self).__init__()
        self.intra_attention = MultiHeadAttention(
            dim_hidden=opt['dim_hidden'],
            num_attention_heads=opt['num_attention_heads'],
            attention_probs_dropout_prob=opt['attention_probs_dropout_prob'],
            hidden_dropout_prob=opt['hidden_dropout_prob'],
            layer_norm_eps=opt['layer_norm_eps'],
            exclude_bias=opt.get('mha_exclude_bias', False)
        )
        self.ffn = PositionwiseFeedForward(
            dim_hidden=opt['dim_hidden'],
            dim_intermediate=opt['intermediate_size'],
            hidden_act=opt['hidden_act'],
            hidden_dropout_prob=opt['hidden_dropout_prob'],
            layer_norm_eps=opt['layer_norm_eps']
        )

    def forward(self, 
            hidden_states: torch.Tensor, 
            attention_mask: Optional[torch.Tensor] = None, 
            head_mask: Optional[torch.Tensor] = None,
            **kwargs
        ) -> Tuple[torch.Tensor, torch.Tensor]:

        hidden_states, attention_probs = self.intra_attention(
            q=hidden_states, 
            k=hidden_states, 
            v=hidden_states,
            input_tensor=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            **kwargs
        )
        hidden_states = self.ffn(hidden_states)
        
        return hidden_states, attention_probs


class DecoderLayer(nn.Module):
    def __init__(self, opt: Dict[str, Any]):
        super(DecoderLayer, self).__init__()
        self.intra_attention = MultiHeadAttention(
            dim_hidden=opt['dim_hidden'],
            num_attention_heads=opt['num_attention_heads'],
            attention_probs_dropout_prob=opt['attention_probs_dropout_prob'],
            hidden_dropout_prob=opt['hidden_dropout_prob'],
            layer_norm_eps=opt['layer_norm_eps'],
            exclude_bias=opt.get('mha_exclude_bias', False)
        )

        if opt.get('fusion', 'temporal_concat') == 'channel_concat':
            dim_key = dim_value = opt['dim_hidden'] * len(opt['modality'])
        else:
            dim_key = dim_value = opt['dim_hidden']

        self.inter_attention = MultiHeadAttention(
            dim_hidden=opt['dim_hidden'],
            dim_key=dim_key,
            dim_value=dim_value,
            num_attention_heads=opt['num_attention_heads'],
            attention_probs_dropout_prob=opt['attention_probs_dropout_prob'],
            hidden_dropout_prob=opt['hidden_dropout_prob'],
            layer_norm_eps=opt['layer_norm_eps'],
            exclude_bias=opt.get('mha_exclude_bias', False)
        )

        self.ffn = PositionwiseFeedForward(
            dim_hidden=opt['dim_hidden'],
            dim_intermediate=opt['intermediate_size'],
            hidden_act=opt['hidden_act'],
            hidden_dropout_prob=opt['hidden_dropout_prob'],
            layer_norm_eps=opt['layer_norm_eps']
        )

    def forward(self, 
            hidden_states: torch.Tensor, 
            encoder_hidden_states: torch.Tensor, 
            attention_mask: Optional[torch.Tensor] = None, 
            encoder_attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            **kwargs
        ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        hidden_states, intra_attention_probs = self.intra_attention(
            q=hidden_states, 
            k=hidden_states, 
            v=hidden_states,
            input_tensor=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            # **kwargs
        )
        hidden_states, inter_attention_probs = self.inter_attention(
            q=hidden_states, 
            k=encoder_hidden_states, 
            v=encoder_hidden_states,
            input_tensor=hidden_states,
            attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            # **kwargs
        )

        hidden_states = self.ffn(hidden_states)
        
        return hidden_states, (intra_attention_probs, inter_attention_probs)
