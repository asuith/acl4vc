''' This module will handle the text generation with beam search. '''
import torch
from models.Beam import Beam
from misc.utils import auto_enlarge, get_shape_and_device

__all__ = ('Translator_ARFormer', )


def get_translator(opt: dict) -> object:
    class_name = 'Translator_{}'.format(opt['decoding_type'])
    if class_name not in globals():
        raise ValueError('We can not find the class `{}` in {}'.format(class_name, __file__))

    return globals()[class_name](opt)


class TranslatorBase(object):
    ''' Load with trained model and handle the beam search '''

    def __init__(self):
        super().__init__()

    def get_inst_idx_to_tensor_position_map(self, inst_idx_list):
        ''' Indicate the position of an instance in a tensor. '''
        return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}
    
    def collect_active_inst_idx_list(self, inst_beams, word_probs, inst_idx_to_position_map):
        ''' Update beams with predicted word probs and collect active (incomplete) beams. '''
        active_inst_idx_list = []
        for inst_idx, inst_position in inst_idx_to_position_map.items():
            is_inst_complete = inst_beams[inst_idx].advance(word_probs[inst_position])

            if not is_inst_complete:
                active_inst_idx_list += [inst_idx]
        return active_inst_idx_list

    def collate_active_info(self, all_inputs_for_decoder, all_decoder_rnn_hidden_states, inst_idx_to_position_map, active_inst_idx_list, beam_size):
        ''' Collect the info of active (incomplete) beams on which the decoder will run. '''
        n_prev_active_inst = len(inst_idx_to_position_map)
        active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
        active_inst_idx = torch.LongTensor(active_inst_idx)
        
        args = (active_inst_idx, n_prev_active_inst, beam_size)

        new_all_inputs_for_decoder = []
        for inputs_for_decoder in all_inputs_for_decoder:
            new_inputs_for_decoder = {}
            for key in inputs_for_decoder.keys():
                new_inputs_for_decoder[key] = self.auto_collect_active_part(inputs_for_decoder[key], *args)
            
            new_all_inputs_for_decoder.append(new_inputs_for_decoder)

        new_all_decoder_rnn_hidden_states = []
        for decoder_rnn_hidden_states in all_decoder_rnn_hidden_states:
            new_all_decoder_rnn_hidden_states.append(self.auto_collect_active_part(decoder_rnn_hidden_states, *args))

        active_inst_idx_to_position_map = self.get_inst_idx_to_tensor_position_map(active_inst_idx_list)
        return new_all_inputs_for_decoder, new_all_decoder_rnn_hidden_states, active_inst_idx_to_position_map
    
    def auto_collect_active_part(self, beamed_tensor, *args):
        ''' Collect tensor parts associated to active beams. '''
        if beamed_tensor is None:
            # this occurs when `beamed_tensor` belongs to the `decoder_rnn_hidden_states` 
            # and the decoder is not based on RNNs
            return None

        if isinstance(beamed_tensor, list):
            if isinstance(beamed_tensor[0], tuple):
                # this occurs when the decoder is multi-layer LSTMs
                # and `beamed_tensor` belongs to the `decoder_rnn_hidden_states` 
                return [
                    tuple([self.collect_active_part(_, *args) for _ in item])
                    for item in beamed_tensor
                ]
            return [self.collect_active_part(item, *args) for item in beamed_tensor]
        else:
            if isinstance(beamed_tensor, tuple):
                # this occurs when the decoder is a one-layer LSTM 
                # and `beamed_tensor` belongs to the `decoder_rnn_hidden_states` 
                return tuple([self.collect_active_part(item, *args) for item in beamed_tensor])
            return self.collect_active_part(beamed_tensor, *args)

    def collect_active_part(self, beamed_tensor, curr_active_inst_idx, n_prev_active_inst, beam_size):
        ''' Collect tensor parts associated to active instances. '''
        _, *d_hs = beamed_tensor.size()
        device = beamed_tensor.device

        n_curr_active_inst = len(curr_active_inst_idx)
        new_shape = (n_curr_active_inst * beam_size, *d_hs)

        beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
        beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx.to(device))
        beamed_tensor = beamed_tensor.view(*new_shape)

        return beamed_tensor

    def collect_hypothesis_and_scores(self, inst_dec_beams, n_best, beam_alpha=1.0):
        all_hyp, all_scores = [], []
        for inst_idx in range(len(inst_dec_beams)):
            scores, tk = inst_dec_beams[inst_idx].sort_finished(beam_alpha)
            n_best = min(n_best, len(scores))
            all_scores += [scores[:n_best]]
            hyps = [inst_dec_beams[inst_idx].get_hypothesis_from_tk(t, k) for t, k in tk[:n_best]]
            all_hyp += [hyps]
        return all_hyp, all_scores


class Translator_ARFormer(TranslatorBase):
    ''' 
        Load with trained model(s) and handle the beam search. 
        Note that model ensembling is available.
    '''
    def __init__(self, opt: dict = {}):
        super().__init__()
        self.beam_size = opt.get('beam_size', 5)
        self.beam_alpha = opt.get('beam_alpha', 1.0)
        self.topk = opt.get('topk', 1)
        self.max_len = opt.get('max_len', 30)

    def prepare_beam_input_ids(self, inst_dec_beams, len_input_ids):
        input_ids = [b.get_current_state() for b in inst_dec_beams if not b.done]
        input_ids = torch.stack(input_ids).to(self.device)
        input_ids = input_ids.view(-1, len_input_ids)
        return input_ids

    def predict_word(self, models, input_ids, all_inputs_for_decoder, all_decoder_rnn_hidden_states, n_active_inst):
        word_probs = []

        new_all_decoder_rnn_hidden_states = []
        for model, inputs_for_decoder, decoder_rnn_hidden_states in \
                zip(models, all_inputs_for_decoder, all_decoder_rnn_hidden_states):
            
            decoding_phase_outputs = model.decoding_phase(
                input_ids=input_ids, 
                inputs_for_decoder=inputs_for_decoder, 
                decoder_rnn_hidden_states=decoder_rnn_hidden_states,
                last_time_step_logits=True
            )
            if 'probs' in decoding_phase_outputs:
                word_probs.append(torch.log(decoding_phase_outputs['probs']))
            else:
                word_probs.append(torch.log_softmax(decoding_phase_outputs['logits'], dim=1))
            new_all_decoder_rnn_hidden_states.append(decoding_phase_outputs.get('decoder_rnn_hidden_states', None))
        
        # average equally
        word_probs = torch.stack(word_probs, dim=0).mean(0) 
        word_probs = word_probs.view(n_active_inst, self.beam_size, -1)
        return word_probs, new_all_decoder_rnn_hidden_states

    def beam_decode_step(self, models, inst_dec_beams, len_input_ids, 
            all_inputs_for_decoder, all_decoder_rnn_hidden_states, inst_idx_to_position_map):
        ''' Decode and update beam status, and then return active beam idx '''

        n_active_inst = len(inst_idx_to_position_map)

        input_ids = self.prepare_beam_input_ids(inst_dec_beams, len_input_ids)
        word_probs, all_decoder_rnn_hidden_states = self.predict_word(models, input_ids, all_inputs_for_decoder, 
                                                                    all_decoder_rnn_hidden_states, n_active_inst)

        # Update the beam with predicted word prob information and collect incomplete instances
        active_inst_idx_list = self.collect_active_inst_idx_list(
            inst_dec_beams, word_probs, inst_idx_to_position_map)

        return active_inst_idx_list, all_decoder_rnn_hidden_states

    def translate_batch(self, models, batch):
        with torch.no_grad():
            all_inputs_for_decoder = []
            all_decoder_rnn_hidden_states = [None] * len(models) # needed for RNN based decoders
            for model in models:
                # handle model ensembling
                encoding_phase_outputs = model.encoding_phase(batch['feats'])
                inputs_for_decoder = model.prepare_inputs_for_decoder(encoding_phase_outputs, batch)
                for key in inputs_for_decoder:
                    # repeat data for beam search
                    inputs_for_decoder[key] = auto_enlarge(inputs_for_decoder[key], self.beam_size)
                
                all_inputs_for_decoder.append(inputs_for_decoder)
            
            (n_inst, *_), self.device = get_shape_and_device(all_inputs_for_decoder[0]['encoder_hidden_states'])
            n_inst //= self.beam_size # because the `encoder_hidden_states` has been enlarged

            #-- Prepare beams
            # TODO: add a variable `candidate_size`?
            inst_dec_beams = [
                Beam(self.beam_size, self.max_len, device=self.device, specific_nums_of_sents=self.topk) 
                for _ in range(n_inst)
            ]

            #-- Bookkeeping for active or not
            active_inst_idx_list = list(range(n_inst))
            inst_idx_to_position_map = self.get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            #-- Decode
            for len_input_ids in range(1, self.max_len):

                active_inst_idx_list, all_decoder_rnn_hidden_states = self.beam_decode_step(
                    models, inst_dec_beams, len_input_ids, 
                    all_inputs_for_decoder, all_decoder_rnn_hidden_states, inst_idx_to_position_map)

                if not active_inst_idx_list:
                    break  # all instances have finished their path to <EOS>

                all_inputs_for_decoder, all_decoder_rnn_hidden_states, inst_idx_to_position_map = self.collate_active_info(
                    all_inputs_for_decoder, all_decoder_rnn_hidden_states, inst_idx_to_position_map, active_inst_idx_list, self.beam_size)
        
        batch_hyps, batch_scores = self.collect_hypothesis_and_scores(inst_dec_beams, self.topk, self.beam_alpha)

        return batch_hyps, batch_scores
