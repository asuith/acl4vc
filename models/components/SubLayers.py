''' Define sublayers in the encoder/decoder layer of Transformer'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from .Attention import ScaledDotProductAttention
from .activations import get_activation
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    def __init__(self, 
            dim_hidden: int, 
            dim_key: Optional[int] = None, 
            dim_value: Optional[int] = None, 
            num_attention_heads: int = 8, 
            attention_probs_dropout_prob: float = 0.0, 
            hidden_dropout_prob: float = 0.5, 
            layer_norm_eps: float = 1e-12,
            exclude_bias: bool = False
        ):
        super(MultiHeadAttention, self).__init__()
        if dim_key is None:
            dim_key = dim_hidden
        if dim_value is None:
            dim_value = dim_hidden

        self.SDPA = ScaledDotProductAttention(dim_hidden, dim_key, dim_value, 
                        num_attention_heads, attention_probs_dropout_prob, exclude_bias)
        self.dense = nn.Linear(dim_hidden, dim_hidden, bias=(not exclude_bias))
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(dim_hidden, eps=layer_norm_eps)
        
    def forward(self, 
            q: torch.Tensor, 
            k: torch.Tensor, 
            v: torch.Tensor, 
            input_tensor: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None, 
            head_mask: Optional[torch.Tensor] = None,
            **kwargs
        ) -> Tuple[torch.Tensor, torch.Tensor]:

        hidden_states, attention_probs = self.SDPA(q, k, v, attention_mask, head_mask, **kwargs)

        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states, attention_probs


class PositionwiseFeedForward(nn.Module):
    def __init__(self, 
            dim_hidden: int, 
            dim_intermediate: int, 
            hidden_act: str = 'relu',
            hidden_dropout_prob: float = 0.5, 
            layer_norm_eps: float = 1e-12,
        ):
        super(PositionwiseFeedForward, self).__init__()
        self.dense1 = nn.Linear(dim_hidden, dim_intermediate)
        self.act = get_activation(hidden_act)
        self.dense2 = nn.Linear(dim_intermediate, dim_hidden)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(dim_hidden, eps=layer_norm_eps)

    def forward(self, hidden_states):
        input_tensor = hidden_states.clone()

        hidden_states = self.dense1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
