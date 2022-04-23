import os

import torch
from pytorch_lightning import LightningModule
from typing import List, Dict, Any, Optional, Tuple, Union

from torch.utils.data.dataloader import DataLoader

from dataloader import get_loader, get_dataset
from models.Framework import get_framework
from misc.crit import get_criterion
from models.Translator import get_translator

import pickle
from tqdm import tqdm
from collections import defaultdict
from misc.cocoeval import COCOScorer, suppress_stdout_stderr
from misc.utils import to_sentence, filter_weight_decay, save_dict_to_csv, Sampler, get_idx_by_sorted_difficulty


class ModelBase(LightningModule):
    def __init__(self, opt: Dict[str, Any], new_opt_used_to_override: Dict[str, Any] = {}):
        super().__init__()
        # passed arguments (hyperparameters) will be saved, we can assess it by `self.hparams`
        self.save_hyperparameters()

        newest_opt = {**self.hparams.opt, **self.hparams.new_opt_used_to_override}

        # captioning model
        self.captioner = get_framework(newest_opt)

        # translator aims to generate captions from scratch with specificed decoding algorithms,
        # e.g., greedy search (beam size = 1), beam search (beam size > 1), etc
        self.translator = get_translator(newest_opt)

        opt = newest_opt
        train_dataset = get_dataset(opt, 'train', print_info=False)
        idx_by_rarity = get_idx_by_sorted_difficulty(train_dataset, caption_difficulty_type=opt["difficulty_type"],
                                                     video_difficulty_path=opt["video_difficulty_path"],
                                                     video_difficulty_weight=opt.get("video_difficulty_weight", 0.3))
        self.sampler = Sampler(idx_by_rarity, opt)
        self.train_dataset = train_dataset
        self.batch_size = opt["batch_size"]
        self.num_workers = opt["num_workers"]

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.sampler.get_sampler(self.trainer.current_epoch,
                                             # self.trainer.logged_metrics.get("Sum", torch.tensor(0)).item(),
                                             self.trainer.logged_metrics.get("Sum", 0),
                                             )
        )


    def training_step(self, batch, batch_idx):
        raise NotImplementedError('Please implement the function `training_epoch_end` in the derived class')
    
    def validation_step(self, batch, batch_idx):
        return self.translate_step(batch, vocab=self.get_vocab(), assert_only_a_caption_per_video=True)
    
    def test_step(self, batch, batch_idx):
        return self.translate_step(batch, vocab=self.get_vocab(), assert_only_a_caption_per_video=True)
    
    def training_epoch_end(self, all_step_outputs) -> None:
        raise NotImplementedError('Please implement the function `training_epoch_end` in the derived class')
    
    def validation_epoch_end(self, all_step_outputs) -> None:
        self.evaluation(all_step_outputs, references=self.get_references(), log_scores=True, log_best_ever_scores=True, verbose=False)

    def test_epoch_end(self, all_step_outputs) -> None:
        scores, detail_scores = self.evaluation(all_step_outputs, references=self.get_references(),
                                                log_scores=True, log_prefix='test', verbose=True)
        # save to csv file
        if self.hparams.opt["save_csv"]:
            path_to_dir = self.hparams.opt["checkpoint_path"]
            save_dict_to_csv(path_to_dir, "test_result.csv", scores)
    
    def forward(self, batch, **kwargs) -> Dict[str, List[dict]]:
        # in lightning, forward defines the prediction/inference actions
        vocab = kwargs.pop('vocab', None)
        if vocab is None:
            vocab = self.get_vocab()
        return self.translate_step(batch, vocab=vocab, **kwargs)
    
    def translate_step(self, 
            batch: Dict[str, Any], 
            vocab: Dict[int, str], 
            assert_only_a_caption_per_video=False, 
            verbose=False,
        ) -> Dict[str, List[dict]]:

        # Model ensembling is achieved by the translator
        if not isinstance(self.captioner, list):
            models = [self.captioner]
        else:
            models = self.captioner

        hyps_of_a_batch, scores_of_a_batch = self.translator.translate_batch(models=models, batch=batch)
        
        bsz = len(hyps_of_a_batch)
        preds_of_a_batch = defaultdict(list)
        for i in range(bsz):
            hyps_of_a_video = hyps_of_a_batch[i]
            scores_of_a_video = scores_of_a_batch[i]
            video_id = batch['video_ids'][i]

            assert isinstance(hyps_of_a_video, list)
            if assert_only_a_caption_per_video:
                assert len(hyps_of_a_video) == 1

            for hyp, score in zip(hyps_of_a_video, scores_of_a_video):
                caption = to_sentence(hyp, vocab)
                
                if verbose:
                    tqdm.write('{}: {}({})'.format(video_id, caption, score))

                preds_of_a_batch[video_id].append({
                    'image_id': video_id, 
                    'caption': caption, 
                    'score': score
                })

        return preds_of_a_batch
    
    def evaluation(self, 
            all_step_outputs: Dict[str, List[dict]], 
            references: Dict[str, List[dict]], 
            scorer: object = COCOScorer(),
            log_scores: bool = True,
            log_best_ever_scores: bool = False,
            log_prefix: str = '',
            verbose: bool = False
        ) -> Tuple[Dict[str, Any], Dict[str, Any]]:

        preds = {}
        for item in all_step_outputs:
            preds.update(item)
        
        with suppress_stdout_stderr():
            scores, detail_scores = scorer.score(references, preds, preds.keys())

        candidate_scores = [scores['Bleu_4'], scores['METEOR'], scores['ROUGE_L'], scores['CIDEr']]
        scores['Sum'] = sum([score for (score, flag) in zip(candidate_scores, self.hparams.opt['metric_sum']) if flag])
        if log_prefix != "test" and self.hparams.opt["sample_method"] != "full" \
                and self.trainer.current_epoch >= self.hparams.opt["sample_fully_t"]:
            for metric, score in scores.items():
                scores[metric] += 100

        if log_prefix != "test" and self.hparams.opt["use_paraphrase"]\
                and self.trainer.current_epoch >= self.hparams.opt["save_after_epoch"]:
            for metric, score in scores.items():
                scores[metric] *= 100

        if log_scores:
            renamed_scores = {'{}_{}'.format(log_prefix, k): v for k, v in scores.items()} if log_prefix else scores
            self.log_dict(renamed_scores)
        
        if log_best_ever_scores:
            if not hasattr(self, 'best_Sum') or scores['Sum'] > self.best_Sum:
                self.best_Sum = scores['Sum']
                self.CIDEr_in_the_best = scores['CIDEr']
            
            if not hasattr(self, 'best_CIDEr') or scores['CIDEr'] > self.best_CIDEr:
                self.best_CIDEr = scores['CIDEr']
            
            self.log('best_Sum', self.best_Sum, prog_bar=True)
            self.log('best_CIDEr', self.best_CIDEr, prog_bar=True)
            # self.log('CIDEr_in_the_best', self.CIDEr_in_the_best, prog_bar=True)
        
        if verbose:
            for k, v in scores.items():
                tqdm.write(k + ': %.4f' % v)
        
        return scores, detail_scores

    def get_vocab(self) -> Dict[int, str]:
        if getattr(self, 'vocab', None) is None:
            self.vocab = pickle.load(open(self.hparams.opt['info_corpus'], 'rb'))['info']['itow']
        return self.vocab
    
    def get_references(self) -> Dict[str, List[dict]]:
        if getattr(self, 'references', None) is None:
            self.references = pickle.load(open(self.hparams.opt['reference'], 'rb'))
        return self.references
    
    def configure_optimizers(self):
        lr = self.hparams.opt.get('learning_rate', 5e-4)
        weight_decay = self.hparams.opt.get('weight_decay', 5e-4)

        if self.hparams.opt.get('filter_weight_decay', False):
            parameters = filter_weight_decay(
                model=self,
                weight_decay=weight_decay,
                filter_biases=self.hparams.opt.get('filter_biases', False),
                skip_list=(),
                skip_substr_list=('memory', )
            )
            optimizer = torch.optim.Adam(parameters, lr=lr)
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters()), lr=lr, weight_decay=weight_decay)

        lr_scheduler_type = self.hparams.opt.get('lr_scheduler_type', 'linear')
        if lr_scheduler_type == 'linear':
            from torch.optim.lr_scheduler import StepLR
            lr_decay = self.hparams.opt.get('lr_decay', 0.9)
            lr_step_size = self.hparams.opt.get('lr_step_size', 1)

            lr_scheduler = StepLR(optimizer, step_size=lr_step_size, gamma=lr_decay)
            other_info = {}
        elif lr_scheduler_type == 'warmup':
            # reference: https://zhuanlan.zhihu.com/p/148487894
            from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
            import math
            lr_decay = self.hparams.opt.get('lr_decay', 0.9)
            lr_step_size = self.hparams.opt.get('lr_step_size', 1)
            epochs = self.hparams.opt.get('epochs', 50)
            milestones = list(range(1, epochs + 1, lr_step_size))
            warm_up_epoch = self.hparams.opt.get('warm_up_epoch', 5)

            # warm_up_with_step_lr
            warm_up_with_multistep_lr = lambda \
                epoch: epoch / warm_up_epoch if epoch <= warm_up_epoch else lr_decay ** len(
                [m for m in milestones if m <= epoch])
            lr_scheduler = LambdaLR(optimizer, lr_lambda=warm_up_with_multistep_lr)

            # # warm_up_with_cosine_lr
            # warm_up_with_cosine_lr = lambda \
            #     epoch: epoch / warm_up_epoch if epoch <= warm_up_epoch else 0.5 * (
            #             math.cos((epoch - warm_up_epoch) / (epochs - warm_up_epoch) * math.pi) + 1)
            # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)

            other_info = {}
        else:
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            lr_decay = self.hparams.opt.get('lr_decay', 0.9)
            lr_monitor_mode = self.hparams.opt.get('lr_monitor_mode', 'max')
            lr_monitor_metric = self.hparams.opt.get('lr_monitor_metric', 'CIDEr')
            lr_monitor_patience = self.hparams.opt.get('lr_monitor_patience', 1)
            min_lr = self.hparams.opt.get('min_lr', 1e-6)

            lr_scheduler = ReduceLROnPlateau(
                optimizer, 
                mode=lr_monitor_mode, 
                factor=lr_decay, 
                patience=lr_monitor_patience,
                min_lr=min_lr
            )
            other_info = {'monitor': lr_monitor_metric, 'strict': True}
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'interval': 'epoch',
                'frequency': 1,
                **other_info
            }
        }
    
    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None) # don't show the version number
        return items
    
    def get_keys_to_device(self, *arg, **kwargs):
        if isinstance(self.captioner, list):
            keys = set()
            for captioner in self.captioner:
                keys = keys | set(captioner.get_keys_to_device(*arg, **kwargs))
            return list(keys)
        else:
            return self.captioner.get_keys_to_device(*arg, **kwargs)


class Model(ModelBase):
    def __init__(self, opt: Dict[str, Any], new_opt_used_to_override: Dict[str, Any] = {}, merge_opt: bool=False):
        if merge_opt:
            opt, new_opt_used_to_override = {**opt, **new_opt_used_to_override}, {}
        super().__init__(opt, new_opt_used_to_override)
        self.criterion = get_criterion(self.hparams.opt) # for training

    def training_step(self, batch, batch_idx):
        # self.current_epoch is automatically provided by `LightningModule`
        feedforward_results = self.captioner(batch, current_epoch=self.current_epoch)
        loss = self.criterion.get_loss({**feedforward_results, **batch})
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log('schedule_sampling_prob', feedforward_results['schedule_sampling_prob'], on_step=True, on_epoch=False, prog_bar=False)
        # save checkpoint
        if self.hparams.opt.get("save_every_10k", False) and self.global_step % 10_000 == 0:
            file_name = f"step_{self.global_step}.ckpt"
            checkpoint_path = self.hparams.opt["checkpoint_path"]
            self.trainer.save_checkpoint(os.path.join(checkpoint_path, "steps", file_name))
        save_at_step = self.hparams.opt.get("save_at_step", -1)
        if save_at_step > 0 and self.global_step  + 1 == save_at_step:  # + 1 is patch for PL bug
            file_name = f"step_{save_at_step}.ckpt"
            checkpoint_path = self.hparams.opt["checkpoint_path"]
            self.trainer.save_checkpoint(os.path.join(checkpoint_path, file_name))
        return loss
        
    def training_epoch_end(self, outputs) -> None:
        loss_info = self.criterion.get_loss_info()
        other_loss_info = {}
        for k in list(loss_info.keys()):
            # too many info for attribute prediction, we do not add them to the bar
            if ('F1-' in k and 'F1-30' not in k) or ('P-' in k):
                other_loss_info[k] = loss_info.pop(k)
        
        self.log_dict(loss_info, prog_bar=True)
        self.log_dict(other_loss_info, prog_bar=False)
        self.criterion.reset_loss_recorder()

    # def train_dataloader(self):
    #     return DataLoader(dataset, shuffle=True, batch_size=64)

    # @staticmethod
    # def add_model_specific_args(parent_parser):
    #     parser = parent_parser.add_argument_group("model")
    #     parser.add_argument('--encoder_layers', type=int, default=12)
    #     parser.add_argument('--data_path', type=str, default='/some/path')
    #     return parent_parser


class ModelEnsemble(ModelBase):
    def __init__(self, 
            checkpoint_paths: List[str], 
            new_opt_used_to_override: Dict[str, Any] = {},
            map_location: Optional[torch.device] = None,
        ):
        '''
            args:
                checkpoint_paths:           a list of pre-trained models' path
                new_opt_used_to_override:   some new arguments, e.g., using a new beam size during inference 
                                            by passing `new_opt_used_to_override` = {'beam_size': 1}
        '''
        assert isinstance(checkpoint_paths, list)
        assert len(checkpoint_paths) >= 1

        all_captioners = []
        opt = None
        decoding_type = None

        for checkpoint_path in checkpoint_paths:
            model = Model.load_from_checkpoint(checkpoint_path, map_location=map_location, strict=True)
            all_captioners.append(model.captioner)
            if opt is None:
                opt = model.hparams.opt

            if decoding_type is None:
                decoding_type = opt['decoding_type']
            else:
                assert decoding_type == model.hparams.opt['decoding_type']
        
        super().__init__(opt, new_opt_used_to_override)
        del self.captioner
        self.captioner = all_captioners
    
    def training_step(self, batch, batch_idx):
        raise NotImplementedError('not supporting the training of multiple models yet!')
        
    def training_epoch_end(self, outputs) -> None:
        raise NotImplementedError('not supporting the training of multiple models yet!')
    
    def train(self) -> None:
        for model in self.captioner:
            model.train()
    
    def eval(self) -> None:
        for model in self.captioner:
            model.eval()
    
    def to(self, device) -> None:
        for model in self.captioner:
            model.to(device)
    
