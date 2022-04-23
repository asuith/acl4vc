import torch
import torch.nn as nn

from typing import List, Dict, Any, Optional

from .Encoder import get_encoder
from .Predictor import get_predictor
from .Decoder import get_decoder 
from .Head import get_cls_head


def get_framework(opt: Dict[str, Any]) -> nn.Module:
    if 'rnn' in opt['decoder'].lower():
        seq2seq_class = RNNSeq2Seq
    else:
        seq2seq_class = TransformerSeq2Seq

    # see the explaination of `input_keys_for_decoder` in the class `Seq2SeqBase` below
    input_keys_for_decoder = ['encoder_hidden_states']
    if opt.get('with_category', False):
        input_keys_for_decoder += ['category']
    # if seq2seq_class is RNNSeq2Seq:
    #    input_keys_for_decoder += ['decoder_rnn_hidden_states']

    # visual_memory_opt = {
    #     'visual_memory_late_fusion': opt.get('visual_memory_late_fusion', False),
    #     'probs_scaler': opt.get('probs_scaler', 0.5)
    # }

    return seq2seq_class(
        encoder=get_encoder(opt),
        predictor=get_predictor(opt),
        decoder=get_decoder(opt),
        cls_head=get_cls_head(opt),
        input_keys_for_decoder=input_keys_for_decoder,
        opt=opt,
    )


class Seq2SeqBase(nn.Module):
    def __init__(self,
            encoder: nn.Module,
            predictor: Optional[nn.Module],
            decoder: nn.Module,
            cls_head: nn.Module,
            input_keys_for_decoder: List[str] = ['encoder_hidden_states'],
            opt: Dict[str, Any] = {},
        ):
        super().__init__()
        ''' 
            Overview of the framework:
                encoder --> decoder --> cls_head
                    |          |
                    ----> predictor 
            
            args:
                encoder:    encodes a video to compact features
                predictor:  to complement some auxiliary tasks,
                            e.g., predicting the sequence length based on the outputs of the encoder
                decoder:    yields hidden states given previsouly generated tokens 
                            and the outputs of the encoder and predictor
                cls_head:   maps the hidden states to logits over the vocabulary
                input_keys_for_decoder: see the explanation below
        '''
        
        self.encoder = encoder
        self.predictor = predictor
        self.decoder = decoder
        self.cls_head = cls_head

        # For convenience, we group decoder inputs into two subgroups
        #   1) previously generated sequences (`input_ids`), which will dynamically change during inference
        #   2) other information like the outputs of the encoder (`encoder_hidden_states`),
        #      auxiliary category information (`category`, optional) etc, which will be 
        #      expanded (repetaed) first and then fixed during inference
        # Herein, we define `input_keys_for_decoder` to specify the inputs of the 2nd subgroup for flexibility
        self.input_keys_for_decoder = input_keys_for_decoder
        '''example (pesudo code from models/Translator.py):

            # before defining `input_keys_for_decoder`
            encoder_hidden_states = auto_enlarge(encoding_phase_outputs['encoder_hidden_states'], beam_size)
            category = auto_enlarge(batch['category'], beam_size)
            some_other_inputs = auto_enlarge(some_other_inputs, beam_size)
            beam_decode_step(..., inst_dec_beams, ..., encoder_hidden_states, category, some_other_inputs, ...)

            # after defining `input_keys_for_decoder`
            inputs_for_decoder = model.prepare_inputs_for_decoder(encoding_phase_outputs, batch)
            for key in inputs_for_decoder:
                inputs_for_decoder[key] = auto_enlarge(inputs_for_decoder[key], beam_size)
            beam_decode_step(..., inst_dec_beams, ..., inputs_for_decoder, ...)
        '''
        self.opt = opt
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                if module.weight.requires_grad:
                    nn.init.xavier_uniform_(module.weight)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                module.weight.data.fill_(1.0)
                module.bias.data.zero_()

        # for p in self.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)
    
    def get_keys_to_device(self, teacher_forcing=False, **kwargs):
        # if we do not use pytorch_lightning.Traniner to automatically manage the device of data
        # we must know which part of data should be moved to specified device
        keys = ['feats']

        if teacher_forcing:
            keys.append('input_ids')

        for k in self.input_keys_for_decoder:
            # exclude intermediate hidden states
            if 'hidden_states' not in k:
                keys.append(k)
        return keys
    
    def encoding_phase(self, feats: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.encoder is not None:
            encoding_phase_outputs = self.encoder(
                input_feats=feats, 
                word_embeddings=self.decoder.get_word_embeddings()
            )
            assert 'encoder_hidden_states' in encoding_phase_outputs.keys()
        else:
            encoding_phase_outputs = {'encoder_hidden_states': feats}
            
        if self.predictor is not None:
            predictor_outputs = self.predictor(**encoding_phase_outputs)
            encoding_phase_outputs.update(predictor_outputs)

        return encoding_phase_outputs
    
    def prepare_inputs_for_decoder(self, 
            encoding_phase_outputs: List[torch.Tensor], 
            batch: Dict[str, Any]
        ) -> Dict[str, torch.Tensor]:
        
        inputs_for_decoder = {}
        for key in self.input_keys_for_decoder:
            if key not in encoding_phase_outputs.keys() and \
                key not in batch.keys():
                raise KeyError('the input key `{}` can not be found in `encoding_phase_outputs` {} \
                    nor `batch` {}'.format(key, encoding_phase_outputs.keys(), batch.keys()))
            
            pointer = batch if key not in encoding_phase_outputs.keys() else encoding_phase_outputs
            inputs_for_decoder[key] = pointer[key]

        return inputs_for_decoder

    def decoding_phase(self,
            input_ids: torch.LongTensor,
            inputs_for_decoder: List[torch.Tensor], 
            last_time_step_logits: bool = False,
            **kwargs
        ) -> Dict[str, torch.Tensor]:
        
        raise NotImplementedError('Please implement this function in `TransformerSeq2Seq` or `RNNSeq2Seq`')
    
    def process_decoding_phase_outputs(self,
            decoding_phase_outputs: Dict[str, torch.Tensor],
            **kwargs
        ) -> Dict[str, torch.Tensor]:
        if self.opt.get('visual_memory_late_fusion', False):
            # reproduce the CVPR 2019 paper: Memory-Attended Recurrent Network for Video Captioning
            assert hasattr(self.decoder, 'visual_memory')
            assert 'logits_m' in decoding_phase_outputs.keys(), \
                'self.decoder.visual_memory should yield `logits_m` (the scores of words in the vocabulary)'
            
            logits_m = decoding_phase_outputs['logits_m']
            
            if self.training:
                # all modules are frozen except self.decoder.visual_memory
                # so the calculation of the caption generation loss only considers `logits_m`
                decoding_phase_outputs['logits'] = logits_m
            else:
                probs_c = torch.softmax(decoding_phase_outputs['logits'], dim=-1)

                if logits_m.dim() != probs_c.dim():
                    # occur when the captioner is the instance of `TransformerSeq2Seq` and `last_time_step_logits` = True
                    assert logits_m.dim() == probs_c.dim() + 1
                    logits_m = logits_m[:, -1, :]
                
                probs_m = torch.softmax(logits_m, dim=-1)

                # follow the paper to balance probability distributions
                scaler = self.opt.get('probs_scaler', 0.5)
                decoding_phase_outputs['probs'] = (1 - scaler) * probs_c + scaler * probs_m

        return decoding_phase_outputs
    
    def feedforward_step(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        # the caption is already known, c.f. batch['input_ids']
        encoding_phase_outputs = self.encoding_phase(batch['feats'])
        inputs_for_decoder = self.prepare_inputs_for_decoder(encoding_phase_outputs, batch)
        
        # for RNN based decoder
        schedule_sampling_prob = 0
        if self.training:
            current_epoch = kwargs.get('current_epoch', None)
            assert current_epoch is not None, 'please pass the arguement `current_epoch`'
            schedule_sampling_saturation_epoch = self.opt.get('schedule_sampling_saturation_epoch', 25)
            schedule_sampling_max_prob = self.opt.get('schedule_sampling_max_prob', 0.25)
            schedule_sampling_prob = min((current_epoch + 1) / schedule_sampling_saturation_epoch, 1.0) * schedule_sampling_max_prob

        decoding_phase_outputs = self.decoding_phase(batch['input_ids'], inputs_for_decoder, 
                                                    schedule_sampling_prob=schedule_sampling_prob, **kwargs)
        
        return {**encoding_phase_outputs, **decoding_phase_outputs, 'schedule_sampling_prob': schedule_sampling_prob}
    
    def forward(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return self.feedforward_step(batch, **kwargs)
    
    
class TransformerSeq2Seq(Seq2SeqBase):
    def decoding_phase(self,
            input_ids: torch.LongTensor,
            inputs_for_decoder: List[torch.Tensor], 
            last_time_step_logits: bool = False,
            **kwargs
        ) -> Dict[str, torch.Tensor]:
            
        decoding_phase_outputs = self.decoder(input_ids, **inputs_for_decoder, **kwargs)
        hidden_states = decoding_phase_outputs['hidden_states']

        if last_time_step_logits:
            logits = self.cls_head(hidden_states[:, -1, :])
        else:
            if not isinstance(hidden_states, list):
                hidden_states = [hidden_states]
            logits = [self.cls_head(item) for item in hidden_states]
        
        decoding_phase_outputs['logits'] = logits
        decoding_phase_outputs = self.process_decoding_phase_outputs(decoding_phase_outputs)
        return decoding_phase_outputs


class RNNSeq2Seq(Seq2SeqBase):
    def scheduled(self, i, sample_mask, item, prob_prev):
        if item is None or prob_prev is None:
            return None
        if sample_mask.sum() == 0:
            it = item[:, i].clone()
        else:
            sample_ind = sample_mask.nonzero().view(-1)
            it = item[:, i].data.clone()
            prob_prev = prob_prev.detach() # fetch prev distribution: shape Nx(M+1)
            it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))

        return it

    def decoding_phase(self,
            input_ids: torch.Tensor,
            inputs_for_decoder: List[torch.Tensor],
            decoder_rnn_hidden_states: Optional[torch.Tensor] = None,
            last_time_step_logits: bool = False,
            **kwargs
        ) -> Dict[str, torch.Tensor]:

        if last_time_step_logits:
            it = input_ids[:, -1] if input_ids.dim() == 2 else input_ids
            decoding_phase_outputs = self.decoder.forward_step(it=it, decoder_rnn_hidden_states=decoder_rnn_hidden_states, **inputs_for_decoder)
            decoding_phase_outputs['logits'] = self.cls_head(decoding_phase_outputs['hidden_states'])
            decoding_phase_outputs = self.process_decoding_phase_outputs(decoding_phase_outputs)
            return decoding_phase_outputs
        
        schedule_sampling_prob = kwargs.pop('schedule_sampling_prob', 0)
        if schedule_sampling_prob == 0:
            # teacher forcing
            decoding_phase_outputs = self.decoder(input_ids, **inputs_for_decoder, **kwargs)
            decoding_phase_outputs['logits'] = self.cls_head(decoding_phase_outputs['hidden_states'])
            decoding_phase_outputs = self.process_decoding_phase_outputs(decoding_phase_outputs)
            return decoding_phase_outputs
        
        # schedule sampling
        hidden_states = []
        logits = []
        attention_probs = []
        decoder_rnn_hidden_states = None

        for i in range(input_ids.size(1)):
            if i >= 1:
                prob = input_ids.new(input_ids.size(0)).float().uniform_(0, 1) # `prob` locates in the same device as input_ids
                mask = prob < schedule_sampling_prob
                it = self.scheduled(i, mask, input_ids, prob_prev=torch.softmax(logits[-1], dim=-1))
            else:
                it = input_ids[:, i]
            
            decoding_phase_outputs = self.decoder.forward_step(
                it=it, 
                decoder_rnn_hidden_states=decoder_rnn_hidden_states, 
                **inputs_for_decoder, 
            )
            decoding_phase_outputs['logits'] = self.cls_head(decoding_phase_outputs['hidden_states'])
            decoding_phase_outputs = self.process_decoding_phase_outputs(decoding_phase_outputs)
            
            hidden_states.append(decoding_phase_outputs['hidden_states'])
            logits.append(decoding_phase_outputs['logits'])
            attention_probs.append(decoding_phase_outputs['attention_probs'])

            decoder_rnn_hidden_states = decoding_phase_outputs['decoder_rnn_hidden_states']
        
        return {
            'hidden_states': torch.stack(hidden_states, dim=1), # [bsz, max_len-1, dim_hidden]
            'logits': torch.stack(logits, dim=1), # [bsz, max_len-1, vocab_size]
            'attention_probs': torch.stack(attention_probs, dim=2), # [bsz, num_feats, max_len-1, n_frames]
        }
