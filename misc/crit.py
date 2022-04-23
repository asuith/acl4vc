from logging import info
from typing import List, Optional, Union, Dict, Tuple
from .logger import AverageMeter
import torch
import torch.nn as nn
from config import Constants
from torch.autograd import Variable
import math
from collections import defaultdict


class CritBase(nn.Module):
    def __init__(self, 
            keys: List[str], 
            weights: Union[List[float], float] = 1.0, 
            batch_mean: bool = True
        ):
        super(CritBase, self).__init__()
        self.keys = keys
        self.weights = weights
        self.batch_mean = batch_mean

    def _step(self, *inputs) -> torch.Tensor:
        raise NotImplementedError()

    def forward(self, kwargs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, float]:
        sources1, sources2, *others = [kwargs[key] for key in self.keys]

        if not isinstance(sources1, list):
            assert type(sources1) is torch.Tensor
            sources1 = [sources1]
        
        if not isinstance(sources2, list):
            assert type(sources2) is torch.Tensor
            sources2 = [sources2] * len(sources1)
        else:
            assert len(sources1) == len(sources2)

        if not isinstance(self.weights, list):
            self.weights = [self.weights] * len(sources1)

        assert len(sources1) == len(self.weights)

        loss = None
        dinominator = float(sources1[0].size(0)) if self.batch_mean else 1.0

        for i, (weight, src1, src2) in enumerate(zip(self.weights, sources1, sources2)):
            if loss is None:
                loss = weight * self._step(i, src1, src2, *others) / dinominator
            else:
                loss = loss + weight * self._step(i, src1, src2, *others) / dinominator
        
        return loss, dinominator


class LanguageGeneration(CritBase):
    def __init__(self, opt):
        visual_word_generation = opt.get('visual_word_generation', False)
        if visual_word_generation:
            weights = opt.get('nv_weights', [0.8, 1.0])
        else:
            weights = 1.0
        super().__init__(keys=['logits', 'labels'], weights=weights)
        self.loss_fn = nn.NLLLoss(reduction='none')
        self.ignore_index = Constants.PAD
        self.num_word_acc = 2 if visual_word_generation else 1
        self.visual_word_generation = visual_word_generation

        self.label_smoothing = opt.get("label_smoothing", 0.0)
        self.temperature = opt.get("with_temperature", 1.0)

    def _step(self, 
            index_indicator: int, 
            logits: torch.Tensor, 
            labels: torch.Tensor, 
            *others
        ):
        """
            args:
                logits: [batch_size, seq_len, vocab_size]
                labels: [batch_size, seq_len]
        """
        assert not len(others)
        assert logits.size(1) == labels.size(1)

        if self.temperature != 1.0:
            logits = logits / self.temperature

        tgt_word_logprobs = torch.log_softmax(logits, dim=-1)

        # calculate the top-1 accuracy of the generated words
        self.calculate_word_acc(index_indicator, tgt_word_logprobs, labels)
        # calculate the perplexity of the generated words
        self.calculate_perplexity(index_indicator, tgt_word_logprobs, labels)

        tgt_word_logprobs = tgt_word_logprobs.contiguous().view(-1, tgt_word_logprobs.size(2))
        labels = labels.contiguous().view(-1)
        loss = (1 - self.label_smoothing) * self.loss_fn(tgt_word_logprobs, labels) + \
               self.label_smoothing * - tgt_word_logprobs.mean(dim=-1)

        if self.ignore_index is not None:
            mask = labels.ne(self.ignore_index).float()
            return torch.sum(loss * mask)
        else:
            return torch.sum(loss)
    
    def calculate_word_acc(self, index_indicator, preds, gts):
        ind = gts.ne(Constants.PAD)
        if index_indicator == 0 and self.visual_word_generation:
            ind = ind & gts.ne(Constants.MASK)
        
        predict_res = preds.max(-1)[1][ind]
        target_res = gts[ind]

        self.word_acc_recorder[index_indicator].update(
                    (predict_res == target_res).sum().item(),
                    predict_res.size(0), 
                    multiply=False
            )

    def calculate_perplexity(self, index_indicator, preds, gts):
        # for the methods with visual word generation
        # we only compute the perplexity of the caption genration process
        if index_indicator == 0 and self.visual_word_generation:
            return None

        assert len(preds.shape) == 3
        assert preds.shape[:-1] == gts.shape

        log_probs = preds.gather(2, gts.unsqueeze(2)).squeeze(2)
        mask = gts.ne(Constants.PAD)
        num_words = float(torch.sum(mask))

        per_word_cross_entropy = -torch.sum(log_probs * mask) / num_words
        self.perplexity_recorder.update(per_word_cross_entropy.item(), num_words)

    def get_fieldsnames(self):
        return ['Word Acc%d' % i for i in range(self.num_word_acc)] + ['Perplexity']

    def get_info(self):
        info = [meter.avg for meter in self.word_acc_recorder]
        info += [math.exp(self.perplexity_recorder.avg)]
        return self.get_fieldsnames(), info

    def reset_recorder(self):
        self.word_acc_recorder = [AverageMeter() for _ in range(self.num_word_acc)]
        self.perplexity_recorder = AverageMeter()


# class AttentionWeightLoss(CritBase):
#     def __init__(self, opt):
#         weight = opt.get("AttentionWeightLoss", 1.0)
#         super().__init__(keys=["labels", "attention_probs", "tgt_visual_taggings"],
#                          weights=weight)
#         self.memory_len = opt.get("memory_len", 8)
#         self.loss_type = opt.get("attention_weight_loss_type", "l2_sum").lower()
#         self.topk_k = opt.get("attention_weight_loss_topk_k", 2)
#         self.ignore_index = Constants.PAD
#         if "l1" in self.loss_type:
#             self.loss_fn = nn.L1Loss(reduction='none')
#         elif "mse" in self.loss_type or "l2" in self.loss_type:
#             self.loss_fn = nn.MSELoss(reduction='none')
#         else:
#             raise NotImplementedError(f"loss_type `{self.loss_type}` is not supported.")

#     def _step(self, index_indicator, labels, preds_attention_probs, tgt_visual_taggings,
#               *others):
#         """
#             args:
#                 labels: [batch_size, num_len]  multi-hot
#                 preds_attention_probs: # [bsz, num_feats, max_len-1, n_frames] num_feats is prob 1.
#                 tgt_visual_taggings: [batch_size, num_len] one-hot, visual word is 1, non-visual is 0.
#         """
#         if self.ignore_index is not None:
#             mask = labels.ne(self.ignore_index).float()
#         else:
#             mask = torch.ones(labels.size(), device=labels.device)
#         if len(preds_attention_probs.shape) >= 5:
#             preds_attention_probs = torch.mean(preds_attention_probs, dim=-1, keepdim=False)
#         preds_attention_probs = preds_attention_probs[:, 0, :, :]

#         seq_len = preds_attention_probs.size(2)
#         visual_nonvisual_index = seq_len - self.memory_len

#         if "sum" in self.loss_type:
#             frame_motion_weights = preds_attention_probs[:, :, :visual_nonvisual_index]
#             weight = torch.sum(frame_motion_weights, dim=-1) * mask
#             tgt_weights = tgt_visual_taggings.type(torch.float) * mask
#         elif "topk" in self.loss_type:
#             memory_weights = preds_attention_probs[:, :, visual_nonvisual_index:]
#             weight, _index = memory_weights.topk(self.topk_k, dim=-1)
#             weight = weight.sum(dim=-1)
#             tgt_weights = (1 - tgt_visual_taggings).type(torch.float) * mask  # non-visual
#         else:
#             raise NotImplementedError(f"loss_type `{self.loss_type}` is not supported.")

#         # calculate loss
#         loss = self.loss_fn(weight, tgt_weights)

#         # update loss values for recorder
#         valid_word_num = mask.sum().item()
#         self.attention_acc_recorder.update(
#             (valid_word_num - torch.abs(weight - tgt_weights).sum().item()) / valid_word_num,
#             1,
#             multiply=False
#         )

#         loss /= valid_word_num
#         return torch.sum(loss)


#     def get_fieldsnames(self):
#         return ['Att Weight Loss']

#     def get_info(self):
#         return self.get_fieldsnames(), [self.attention_acc_recorder.avg]

#     def reset_recorder(self):
#         self.attention_acc_recorder = AverageMeter()


# class NoisyOrMIL(CritBase):
#     def __init__(self, opt):
#         super().__init__(keys=['preds_attr', 'labels_attr'])
#         self.crit_type = opt.get('attr_crit', 'focal')
#         self.alpha, self.gamma = opt.get('attr_focal_args', [0.95, 2.0])
#         self.topk_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
#         self.wise = opt.get('attribute_wise', False)
#         self.num_modalities = len(opt['modality'])
#         self.names = opt['modality'].upper()

#     def _step(self, index_indicator, preds_attr, labels_attr, *others):
#         """
#             args:
#                 preds_attr:     [batch_size, n_attributes]
#                 labels_attr:    [batch_size, n_attributes], multi-hot
#         """
#         assert not len(others)
#         assert preds_attr.shape[1] <= labels_attr.shape[1]
#         labels_attr = labels_attr[:, :preds_attr.shape[1]]
        
#         preds_attr = torch.clamp(preds_attr, 0.01, 0.99)

#         bce = -(labels_attr * torch.log(preds_attr) + (1.0 - labels_attr) * torch.log(1.0 - preds_attr))
#         device, _dtype = preds_attr.device, preds_attr.dtype

#         if 'focal' in self.crit_type:
#             alpha_factor = torch.tensor(self.alpha).to(device)
#             gamma = torch.tensor(self.gamma).to(device)

#             alpha_factor = torch.where(torch.eq(labels_attr, 1.), alpha_factor, 1. - alpha_factor)
#             focal_weight = torch.where(torch.eq(labels_attr, 1.), 1. - preds_attr, preds_attr)
#             focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
            
#             loss = focal_weight * bce # [batch_size, n_attributes]
#         else:
#             loss = bce

#         # avoid zero division
#         if 'sum' in self.crit_type:
#             loss = loss.sum(1)
#         else:
#             mininal_dinominator = torch.tensor(1.0).to(device)
#             loss = loss.sum(1) / torch.max(mininal_dinominator, labels_attr.sum(1).float())
        
#         if getattr(self, 'f1_recorder', None) is not None:
#             _, candidates = preds_attr.topk(max(self.topk_list), dim=1, sorted=True, largest=True)
#             total_n_positive = labels_attr.sum(1)

#             for i, topk in enumerate(self.topk_list):
#                 this_candidates = candidates[:, :topk]
#                 this_n_hit = labels_attr.gather(1, this_candidates).sum(1)
#                 this_n_hit[this_n_hit.eq(0)] = 1e-3
#                 precision = this_n_hit / topk
#                 recall = this_n_hit / total_n_positive
#                 f1 = 2 * precision * recall / (precision + recall)
#                 if self.wise:
#                     self.f1_recorder[index_indicator][i].update(f1.sum().item(), f1.size(0), multiply=False)
#                 else:
#                     self.f1_recorder[i].update(f1.sum().item(), f1.size(0), multiply=False)
#                     self.precision_recorder[i].update(precision.sum().item(), precision.size(0), multiply=False)

#         return torch.sum(loss).float()
    
#     def get_fieldsnames(self):
#         if self.wise:
#             fieldsnames = []
#             for char in self.names:
#                 fieldsnames += ['F1-%s-%02d'%(char, item) for item in self.topk_list]
#             return fieldsnames
#         else:    
#             return ['F1-%02d'%item for item in self.topk_list] + ['P-%02d'%item for item in self.topk_list]

#     def get_info(self):
#         if self.wise:
#             data = []
#             for i in range(self.num_modalities):
#                 data += [item.avg for item in self.f1_recorder[i]]
#             return self.get_fieldsnames(), data
#         else:
#             return self.get_fieldsnames(), [item.avg for item in self.f1_recorder] + [item.avg for item in self.precision_recorder]

#     def reset_recorder(self):
#         if self.wise:
#             self.f1_recorder = [[AverageMeter() for _ in range(len(self.topk_list))] for __ in range(self.num_modalities)]
#         else:
#             self.f1_recorder = [AverageMeter() for _ in range(len(self.topk_list))]
#             self.precision_recorder = [AverageMeter() for _ in range(len(self.topk_list))]


class Criterion(object):
    """
        Calculating losses or some metrics for all tasks

        Standard operations:
            1. before a epoch, Criterion.reset_loss_recorder()
            2. during a epoch, Criterion.get_loss(forward_results)
            3. after  a epoch, Criterion.get_loss_info()
    """ 
    def __init__(self, crit_objects, names, scales, summarywriter=None):
        assert len(crit_objects) == len(names)
        assert len(names) == len(scales)
        self.crit_objects = crit_objects
        self.num_loss = len(crit_objects)
        self.names = names
        self.scales = scales
        self.summarywriter = summarywriter
        self.n_current_round = 0

        self.reset_loss_recorder()
        
    def reset_loss_recorder(self):
        self.loss_recorder = [AverageMeter() for _ in range(self.num_loss)]
        for crit_object in self.crit_objects:
            if getattr(crit_object, 'reset_recorder', None) is not None:
                crit_object.reset_recorder()

    def get_loss(self, results, **kwargs):
        """
            args:
                results: dict, contains the forward results of the model and some ground-truths
        """
        loss = []
        for i in range(self.num_loss):
            # calculate the i-th loss
            assert isinstance(self.crit_objects[i], CritBase)
            i_loss, num_samples = self.crit_objects[i](results)
            
            # weighting the i-th loss
            loss.append(i_loss * self.scales[i])

            # update the statistics of the i-th loss
            self.loss_recorder[i].update(i_loss.item(), num_samples)

        # loss = loss1 * scale1 + loss2 * scale2 + ... 
        loss = torch.stack(loss, dim=0).sum(0)
        return loss

    def get_loss_info(self):
        all_names = self.names.copy()
        all_info = [meter.avg for meter in self.loss_recorder]

        for crit_object in self.crit_objects:
            if getattr(crit_object, 'get_info', None) is not None:
                this_name, this_info = crit_object.get_info()
                all_names += this_name
                all_info += this_info

        if self.summarywriter is not None:
            self.n_current_round += 1
            for name, loss in zip(all_names, all_info):
                self.summarywriter.add_scalar(name, loss, global_step=self.n_current_round)

        # e.g., ['Cap Loss', 'Word Acc0', 'Perplexity'], [31.8, 0.385, 53.0]
        # return all_names, all_info 
        return {n: i for n, i in zip(all_names, all_info)}
    
    def get_fieldsnames(self):
        exclude_index_set = []
        fieldsnames = []
        for i, crit_object in enumerate(self.crit_objects):
            if isinstance(crit_object, LanguageGeneration):
                exclude_index_set.append(i)
            elif getattr(crit_object, 'get_fieldsnames', None) is not None:
                fieldsnames += crit_object.get_fieldsnames()

        fieldsnames += [n for i, n in enumerate(self.names) if i not in exclude_index_set]                
        return fieldsnames


def get_criterion(opt, summarywriter=None):
    assert isinstance(opt['crit'], list)

    crit_objects = []
    for item in opt['crit']:
        crit_name = item.lower()
        if crit_name == 'lang':
            this_crit_object = LanguageGeneration(opt)
        # elif crit_name == 'length':
        #     this_crit_object = nn.KLDivLoss()
        elif crit_name == 'attention_weight':
            this_crit_object = AttentionWeightLoss(opt)
        elif crit_name == 'attribute':
            this_crit_object = NoisyOrMIL(opt)
        else:
            raise NotImplementedError('''Please make sure that:\n
                1) the coressponding criterion for \'{}\' has been implemented in misc.crit;\n
                2) add \"elif crit_name == \'{}\': this_crit_object = xxx\" in misc.crit.get_criterion().\n
                '''.format(crit_name, crit_name))

        crit_objects.append(this_crit_object)

    return Criterion(
            crit_objects=crit_objects,
            names=opt['crit_name'],
            scales=opt['crit_scale'],
            summarywriter=summarywriter
        )

