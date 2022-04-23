'''
    Author: Bang Yang
'''
# import warnings
# warnings.filterwarnings('ignore')
import os
import torch

from misc.utils import save_dict_to_csv
from models import ModelEnsemble, Model
from dataloader import get_loader
from pytorch_lightning import Trainer
import argparse
from tqdm import tqdm
from typing import Union

from misc.utils import to_device
from misc.visual_memory import (
    run_visual_memory, 
    add_visual_memory_specific_args
)


def run_eval(
        args, 
        model: Union[ModelEnsemble, Model], 
        loader: torch.utils.data.DataLoader, 
        device: torch.device
    ):
    model.eval()
    model.to(device)
    
    vocab = model.get_vocab()
    references = model.get_references()

    all_step_outputs = []
    for batch in tqdm(loader):
        with torch.no_grad():
            for k in model.get_keys_to_device():
                if k in batch:
                    batch[k] = to_device(batch[k], device)

            step_outputs = model.translate_step(
                batch=batch,
                vocab=vocab,
                assert_only_a_caption_per_video=True,
                verbose=args.verbose,
            )
        all_step_outputs.append(step_outputs)
    
    scores, detail_scores = model.evaluation(
        all_step_outputs=all_step_outputs,
        references=references,
        verbose=True,
        log_prefix='test',
    )
    scores["category"] = args.specific
    if args.save_csv:
        if len(args.checkpoint_paths) == 1:
            path_to_dir = os.path.split(args.checkpoint_paths[0])[0]
        else:
            base_dir = os.path.split(os.path.split(os.path.split(args.checkpoint_paths[0])[0])[0])[0]
            models_with_scope = ["-".join(cp.split(base_dir)[-1].split("/")[1:-1]) for cp in args.checkpoint_paths]
            path_to_dir = os.path.join(base_dir, "Ensemble", "_+_".join(sorted(models_with_scope)))
            os.makedirs(path_to_dir, exist_ok=True)
        save_dict_to_csv(path_to_dir, "test_result.csv", scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='translate.py')
    parser.add_argument('-cp', '--checkpoint_paths', type=str, nargs='+', required=True)

    cs = parser.add_argument_group(title='Common Settings')
    cs.add_argument('-gpus', '--gpus', type=int, default=1)
    cs.add_argument('-fast', '--fast', default=False, action='store_true', 
        help='directly use Trainer.test()')
    cs.add_argument('-v', '--verbose', default=False, action='store_true',
        help='print some intermediate information (works when `fast` is False)')
    cs.add_argument('--save_csv', default=False, action='store_true',
        help='save result to csv file in model path (works when `fast` is False)')

    ds = parser.add_argument_group(title='Dataloader Settings')
    ds.add_argument('-bsz', '--batch_size', type=int, default=128)
    ds.add_argument('-mode', '--mode', type=str, default='test',
        help='which set to run?', choices=['train', 'validate', 'test', 'all'])
    ds.add_argument('-specific', '--specific', default=-1, type=int, 
        help='run on the data of the specific category (only works in the MSR-VTT)')

    ar = parser.add_argument_group(title='Autoregressive Decoding Settings')
    ar.add_argument('-bs', '--beam_size', type=int, default=5, help='Beam size')
    ar.add_argument('-ba', '--beam_alpha', type=float, default=1.0)
    ar.add_argument('-topk', '--topk', type=int, default=1)

    na = parser.add_argument_group(title='Non-Autoregressive Decoding Settings')
    na.add_argument('-i', '--iterations', type=int, default=5)
    na.add_argument('-lbs', '--length_beam_size', type=int, default=6)
    na.add_argument('-q', '--q', type=int, default=1)
    na.add_argument('-qi', '--q_iterations', type=int, default=1)
    na.add_argument('-paradigm', '--paradigm', type=str, default='mp', choices=['mp', 'ef', 'l2r'])
    na.add_argument('-use_ct', '--use_ct', default=False, action='store_true')
    na.add_argument('-md', '--masking_decision', default=False, action='store_true')
    na.add_argument('-ncd', '--no_candidate_decision', default=False, action='store_true')
    na.add_argument('--algorithm_print_sent', default=False, action='store_true')

    ts = parser.add_argument_group(title='Task Settings')
    ts.add_argument('-latency', '--latency', default=False, action='store_true', 
        help='batch_size will be set to 1 to compute the latency, which will be saved to latency.txt in the checkpoint folder')
    ts.add_argument('-vm', '--visual_memory', default=False, action='store_true',
        help='construct the visual memory, i.e., <word, (word emb, visual context)> pairs')
    
    parser.add_argument('-json_path', type=str, default='')
    parser.add_argument('-json_name', type=str, default='')
    parser.add_argument('-ns', '--no_score', default=False, action='store_true')
    parser.add_argument('-analyze', default=False, action='store_true')
    parser.add_argument('-collect_path', type=str, default='./collected_captions')
    parser.add_argument('-collect', default=False, action='store_true')
    parser.add_argument('-nobc', '--not_only_best_candidate', default=False, action='store_true')

    parser = add_visual_memory_specific_args(parser)
    args = parser.parse_args()
    
    if args.fast:
        # fast mode
        model = ModelEnsemble(args.checkpoint_paths, vars(args))
        trainer = Trainer(logger=False, gpus=args.gpus)
        opt = model.hparams.opt
        loader = get_loader(opt, args.mode, print_info=True, specific=args.specific, 
            not_shuffle=True, batch_size=args.batch_size
        )
        trainer.test(model, loader)
    else:
        # handle the device and running loop on your own
        device = torch.device('cpu' if args.gpus == 0 else 'cuda')

        if len(args.checkpoint_paths) == 1:
            model = Model.load_from_checkpoint(
                args.checkpoint_paths[0], 
                new_opt_used_to_override=vars(args),
                map_location=device,
            )
        else:
            # `load_from_checkpoint` is called in `ModelEnsemble.__init__()` for each checkpoint
            model = ModelEnsemble(args.checkpoint_paths, vars(args), map_location=device)

        if args.visual_memory:
            run_visual_memory(args, model, device)
        else:
            loader = get_loader(model.hparams.opt, args.mode, print_info=True, specific=args.specific, 
                not_shuffle=True, batch_size=args.batch_size, is_validation=True
            )
            run_eval(args, model, loader, device)

'''
python train.py --scope test --gpus 0 --epochs 1 --num_workers 0 --decoder SingleLayerRNNDecoder
python train.py --scope dummy --gpus 0 --epochs 1 --num_workers 0 --decoder TransformerDecoder
python translate.py --mode train --memory -cp ./experiments/MSRVTT/test/best.ckpt --gpus 0
python translate.py --mode train --memory -cp ./experiments/MSRVTT/dummy/best.ckpt --gpus 0

CUDA_VISIBLE_DEVICES=0 python train.py --gpus 1 --encoder Encoder_NaiveLN --decoder SingleLayerRNNDecoder --scope LN_LSTM --num_workers 4 --weight_decay 0.001
CUDA_VISIBLE_DEVICES=0 python translate.py -cp ./experiments/MSRVTT/LN_LSTM/best.ckpt -vm -vm_topk 10 -vm_topk_per_video 1
CUDA_VISIBLE_DEVICES=0 python translate.py -cp ./experiments/MSRVTT/LN_LSTM/best.ckpt -vm -vm_topk 10 -vm_topk_per_video 1 -vm_use_scores

CUDA_VISIBLE_DEVICES=0 python translate.py -cp ./experiments/MSRVTT/LN_LSTM/best.ckpt --path_to_load_videos ~/new_VC_data/MSRVTT/all_videos -vm -vme_topk 15 -vme_word stroller
CUDA_VISIBLE_DEVICES=0 python translate.py -cp ./experiments/MSRVTT/LN_LSTM/best.ckpt --path_to_load_videos ~/new_VC_data/MSRVTT/all_videos -vm -vm_use_scores -vme_topk 15 -vme_word stroller
'''
