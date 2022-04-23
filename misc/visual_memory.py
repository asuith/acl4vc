import os
import numpy as np
import pickle
from tqdm import tqdm
from typing import Any, Dict, List, Tuple, Union
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from dataloader import get_loader
from models.Wrapper import ModelEnsemble, Model
from models.Framework import TransformerSeq2Seq
from config import Constants
from .utils import to_device


def add_visual_memory_specific_args(parent_parser: object) -> object:
    parser = parent_parser.add_argument_group(title='Visual Memory Settings (with the `--memory` argument)')
    parser.add_argument('-vm_topk_max', '--visual_memory_topk_max', type=int, default=500, 
            help='the maximun number of relevant visual content for words when generating vid2relevant '
            '(preliminary of visual memory generation, avoid repeative genration)')
    
    parser.add_argument('-vm_topk_per_video', '--visual_memory_topk_per_video', type=int, default=1, 
            help='the number of unique relevant visual content per video when generating vid2relevant')
    
    parser.add_argument('-vm_use_scores', '--visual_memory_use_scores', default=False, action='store_true',
            help='use attention scores rather than attention probs when generating vid2relevant')

    parser.add_argument('-vm_topk', '--visual_memory_topk', type=int, default=10, 
            help='the number of relevant visual content for words when generating memory '
            '(based on the pre-generated vid2relevant, '
            '`memory_topk` should not be larger than `visual_memory_topk_max`)')
    
    parser.add_argument('-vm_source_type', '--visual_memory_source_type', type=str, default='hidden', 
            help='which features to constitute visual contexts, either `raw` (features from backbone CNNs) or '
            '`hidden` (default, features output from the encoder of the captioner)', 
            choices=['raw', 'hidden'])
    
    parser.add_argument('-vm_fusion_type', '--visual_memory_fusion_type', type=str, default='add', 
            help='if there are several modalities, which way to fuse them? either '
            '`add` (default, same dimension should be guaranteed) or `concat`', 
            choices=['add', 'concat'])
    
    parser.add_argument('-vme_word', '--visual_memory_example_word', type=str, default='', 
            help='show the most relevant visual content to the specific word (default to None); '
            'if specifying a word, other functions will not be called (e.g., generating memory)')
    
    parser.add_argument('-vme_topk', '--visual_memory_example_topk', type=int, default=5, 
            help='the number of the most relevant visual content to the specific word (default to 5)')
    
    parser.add_argument('-vme_save_path', '--visual_memory_example_save_path', type=str, default='visualization/visual_memory')
    parser.add_argument('-vme_save_name', '--visual_memory_example_save_name', type=str, default='')

    parser.add_argument('--path_to_load_videos', type=str, default='',
            help='The path to video files when visualizing an example; '
            'by default, it will be set to os.path.join(Constant.base_data.path, dataset, all_videos)')
    
    parser.add_argument('--video_suffix', type=str, default='mp4')
    return parent_parser


def get_wid2pos_and_lengths_and_vids(
        loader: DataLoader
    ) -> Tuple[Dict[int, List[int]], List[int], np.ndarray]:

    cap_lengths = []
    video_ids = []
    wid2pos = defaultdict(list)
    pos_till_now = 0

    for batch in tqdm(loader):
        labels = batch['labels'].cpu().numpy()

        for i in range(labels.shape[0]):
            for j, wid in enumerate(labels[i]):
                this_pos = pos_till_now + j
                wid2pos[wid].append(this_pos)
                if wid == Constants.EOS:
                    length_of_this_cap = j + 1
                    cap_lengths.append(length_of_this_cap)
                    pos_till_now += length_of_this_cap
                    # e.g. the length of this caption is 3, then 'video123' is changed to [123, 123, 123]
                    vid = [int(batch['video_ids'][i][5:])] * length_of_this_cap
                    video_ids.extend(vid)
                    break
    
    return wid2pos, cap_lengths, np.array(video_ids)
                

def generate_attention_maps(
        model: Union[ModelEnsemble, Model], 
        loader: DataLoader,
        device: torch.device,
        cap_lengths: List[int],
        attention_maps_path: str,
        return_attention_scores: bool = False
    ) -> None:
    
    attention_maps = []
    is_transformer = isinstance(model.captioner, TransformerSeq2Seq)
    
    global_i = 0
    for batch in tqdm(loader):
        with torch.no_grad():
            for k in model.get_keys_to_device(teacher_forcing=True):
                if k in batch:
                    batch[k] = to_device(batch[k], device)

            step_outputs = model.captioner.feedforward_step(batch, return_attention_scores=return_attention_scores)
            if is_transformer:        
                # temporal concat: [bsz, n_heads, seq_len, n_frames * n_modality]
                attention_probs = step_outputs['all_inter_attentions'][-1] # the last decoder layer
            else:
                # temporal concat: [bsz, 1, seq_len, n_frames * n_modality]
                attention_probs = step_outputs['attention_probs'] 
            
            attention_probs = attention_probs.mean(1).cpu()
            bsz = batch['labels'].size(0)
            for i in range(bsz):
                length_of_this_cap = cap_lengths[global_i]
                trunc_attention_probs = attention_probs[i, :length_of_this_cap, :] # [length, n_frames * n_modality]
                attention_maps.append(trunc_attention_probs)
                global_i += 1
    
    attention_maps = torch.cat(attention_maps, dim=0) # [sum(cap_lengths), n_frames * n_modality]
    np.save(attention_maps_path, attention_maps)
    

def generate_wid2relevant(
        n_topk: int, 
        n_topk_per_video: int,
        attention_maps: np.ndarray, 
        wid2pos: Dict[int, List[int]], 
        video_ids: np.ndarray, 
        n_frames: int, 
        root: str, 
        filenames: List[str]
    ):
    assert len(attention_maps.shape) == 2
    assert attention_maps.shape[0] == len(video_ids)

    for i in range(0, attention_maps.shape[1], n_frames):
        wid2relevant = {}

        attention_maps_of_this_modality = attention_maps[:, i:i+n_frames] # [sum(cap_lengths), n_frames]
        for wid, pos in tqdm(wid2pos.items()):
            # if str(wid) in db.keys():
            #     continue

            video_ids_of_this_wid = video_ids[pos] # [len(pos)]
            attention_maps_of_this_wid = attention_maps_of_this_modality[pos] # [len(pos), n_frames]
            attention_maps_of_this_wid = torch.from_numpy(attention_maps_of_this_wid).view(-1)

            probs, indices = attention_maps_of_this_wid.sort(descending=True)
            valid_n_topk = min(attention_maps_of_this_wid.shape[0], n_topk)
            vid_record = {}
            indice_record = set()
            new_probs, new_indices = [], []
            # get unique valid_n_topk pairs
            for j, (p, indice) in enumerate(zip(probs, indices)):
                index = indice // n_frames
                offset = indice % n_frames # frame_id
                vid = video_ids_of_this_wid[index] # type of int (without prefix `video`)

                new_indice = vid * n_frames + offset

                vid_record[vid] = vid_record.get(vid, 0) + 1
                if vid_record[vid] <= n_topk_per_video:
                    if new_indice in indice_record: # same vid, same frame id
                        continue
                    
                    indice_record.add(new_indice)
                    
                    new_probs.append(p)
                    new_indices.append(new_indice)
                    if len(new_probs) >= valid_n_topk:
                        break
            
            # print(new_indices[:10]) 
            wid2relevant[wid] = np.array([new_probs, new_indices], dtype=np.float32)

        filename = filenames[i // n_frames]
        with open(os.path.join(root, filename), 'wb') as f:
            pickle.dump(wid2relevant, f)


def get_preliminary(
        args: object, 
        model: Union[ModelEnsemble, Model], 
        device: torch.device,
    ) -> Tuple[str, Dict[int, np.ndarray]]:
    assert len(args.checkpoint_paths) == 1

    root = os.path.dirname(args.checkpoint_paths[0])
    measure_type = 'scores' if args.visual_memory_use_scores else 'probs'
    file_field = '{}_{}pv'.format(measure_type, args.visual_memory_topk_per_video)

    print('- Checking wheter wid2relevant has been generated or not:')
    basename = 'wid2relevant_{}'.format(file_field)
    if model.hparams.opt['fusion'] == 'temporal_concat':
        filenames = [basename + '_{}.pkl'.format(char) for char in model.hparams.opt['modality']]
    else:
        filenames = [basename + '.pkl']
    
    done_flag = True
    for fn in filenames:
        path = os.path.join(root, fn)
        if not os.path.exists(path):
            print('- {} does not exist!'.format(path))
            done_flag = False
        else:
            print('- {} exists!'.format(path))
    
    if done_flag:
        print('- wid2relevant has been generated in {}'.format(root))
    else:
        print('- Start generating wid2relevant:')
    
        loader = get_loader(model.hparams.opt, mode='train', print_info=True,
            not_shuffle=True, batch_size=args.batch_size, is_validation=True, all_caps=True
        )

        print('- Step 1: preparing wid2pos, cap_lengths and video_ids')
        wid2pos, cap_lengths, video_ids = get_wid2pos_and_lengths_and_vids(loader)

        print('- Step 2: preparing attention maps of all training videos')
        attention_maps_fn = 'attention_maps_{}.npy'.format(measure_type)
        attention_maps_path = os.path.join(root, attention_maps_fn)
        if not os.path.exists(attention_maps_path):
            print('- We can not find attention maps in {}'.format(attention_maps_path))
            print('- Start generating ...')
            generate_attention_maps(model, loader, device, cap_lengths, attention_maps_path, return_attention_scores=args.visual_memory_use_scores)
            print('- Done!')
        else:
            print('- Loading pre-extracted attention maps in {}'.format(attention_maps_path))

        attention_maps = np.load(attention_maps_path)
        assert len(attention_maps.shape) == 2
        assert attention_maps.shape[0] == len(video_ids)

        print('- Step 3: finding most relevant {} frames/segments for each word'.format(args.visual_memory_topk_max))
        generate_wid2relevant(
            n_topk=args.visual_memory_topk_max, 
            n_topk_per_video=args.visual_memory_topk_per_video,
            attention_maps=attention_maps, 
            wid2pos=wid2pos, 
            video_ids=video_ids, 
            n_frames=model.hparams.opt['n_frames'], 
            root=root, 
            filenames=filenames
        )
    
    wid2relevant = []
    for fn in filenames:
        wid2relevant.append(pickle.load(open(os.path.join(root, fn), 'rb')))

    return root, wid2relevant, file_field


def plot_visual_memory_example(
        wid2relevant: Dict[int, np.ndarray], 
        word: str, 
        topk: int, 
        vocab: Dict[int, str], 
        path_to_load_videos: str, 
        video_suffix: str, 
        n_frames: int, 
        save_path: str = ''
    ) -> None:
    # packages for visualizing examples
    from pretreatment.extract_frames_from_videos import extract_frames
    from misc.utils import get_ids_of_keyframes
    from glob import glob
    from PIL import Image
    import matplotlib.pyplot as plt
    import shutil

    word2wid = {v: k for k, v in vocab.items()}
    assert word in word2wid.keys(), \
        'Sorry, the specified word `{}` can not be found in the vocabulary'.format(word)
    
    assert word not in [Constants.PAD_WORD, Constants.BOS_WORD, Constants.MASK_WORD, Constants.VIS_WORD]

    wid = word2wid[word]
    relevant = wid2relevant[wid] # [2, topk]
    valid_topk = min(topk, relevant.shape[1])
    relevant_probs, relevant_indices = relevant[:, :valid_topk]
    print(relevant_indices)

    # sample frames from relevant videos first and then load relevant images (frames)
    frames_path = './tmp_visualizing_visual_memory_of_a_word'
    images = []
    for i, indice in enumerate(relevant_indices):
        vid = 'video{}'.format(int(indice) // n_frames)
        frame_id = int(indice) % n_frames
        print('- {}: vid({}), fid({}), prob({:.4f})'.format(i, vid, frame_id, relevant_probs[i]))

        path_to_the_video = os.path.join(path_to_load_videos, '{}.{}'.format(vid, video_suffix))
        assert os.path.exists(path_to_the_video), \
            'Sorry, we can not find the video in {}'.format(path_to_the_video)
        
        # frames will be saved as vid_00001.png vid_00002.png ...
        extract_frames(
            video=path_to_the_video,
            dst=frames_path,
            prefix='{}_'.format(vid),
            suffix='png',  
            cleanup=False,
            strategy=0 # extract all frames of the video
        )

        n_total_frames = len(glob(os.path.join(frames_path, vid)))
        ids = get_ids_of_keyframes(
            total_frames_of_a_video=n_total_frames,
            k=n_frames,
            identical=True,
            offset=1 # the first sampled frame is vid_00001.png (start from 1 rather than 0)
        )

        image_name = '{}_{:05d}.png'.format(vid, ids[frame_id])
        image = Image.open(os.path.join(frames_path, image_name))
        images.append(image)

    # visualize
    fig = plt.figure(dpi=300)
    for i in range(valid_topk):
        ax = plt.subplot(1, valid_topk, i + 1)
        ax.imshow(images[i])
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title('Prob: {:.4f}'.format(relevant_probs[i]))
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

    # cleanup
    shutil.rmtree(frames_path)


def generate_visual_memory(
        wid2relevant: Dict[int, np.ndarray], 
        model: Union[ModelEnsemble, Model], 
        device: torch.device,
        topk: int, 
        root: str,
        file_field: str,
        source_type: str = 'hidden',
        fusion_type: str = 'add',
    ) -> None:
    opt = model.hparams.opt

    assert opt['fusion'] == 'temporal_concat'

    loader = get_loader(opt, mode='train', print_info=True,
        not_shuffle=True, is_validation=True, all_caps=False
    )

    # prepare features of all modalities
    feats = defaultdict(list)
    modality = opt['modality']
    
    for batch in tqdm(loader):
        if source_type == 'raw':
            # use features extracted from backbone CNNs
            this_feats = batch['feats']
        else:
            for k in model.get_keys_to_device(teacher_forcing=True):
                if k in batch:
                    batch[k] = to_device(batch[k], device)

            with torch.no_grad():
                step_outputs = model.captioner.feedforward_step(batch)

            if opt['fusion'] == 'temporal_concat':
                # [bsz, n_frames * n_modality, d]
                this_feats = torch.chunk(step_outputs['encoder_hidden_states'], chunks=len(modality), dim=1)
            elif opt['fusion'] == 'none':
                this_feats = step_outputs['encoder_hidden_states']
            else:
                raise NotImplementedError('We only support `temporal_concat` and `none`')

        for char, f in zip(modality, this_feats):
            key = 'feats_%s' % char
            feats[key].append(f)

    for k in feats.keys():
        feats[k] = torch.cat(feats[k], dim=0).cpu().contiguous()
        
    batch = feats
    assert len(modality) == len(wid2relevant)

    fused_memory_path = os.path.join(root, 'memory_{}_fused_top{}_{}.npy'.format(file_field, topk, source_type))
    if os.path.exists(fused_memory_path):
        print('- The fused memory has been saved in {}'.format(fused_memory_path))
        return None
    
    all_memory = []
    for char, w2r in zip(modality, wid2relevant):
        feats = batch['feats_{}'.format(char)] # [n_training_videos, n_frames, dim]
        feats = feats.view(-1, feats.size(-1)) # [n_training_videos * n_frames, dim]

        memory = np.zeros((opt['vocab_size'], feats.size(-1)))
        
        for wid, (probs, indices) in w2r.items():
            topk_probs = torch.from_numpy(probs[:topk])
            topk_indices = indices[:topk]
            topk_feats = feats[topk_indices, :]

            result = torch.matmul(topk_probs.unsqueeze(0), topk_feats) / topk_probs.sum() # [1, topk] * [topk, dim] = [1, dim]
            memory[wid, :] = result.squeeze(0).numpy()

        all_memory.append(memory)

        save_path = os.path.join(root, 'memory_{}_{}_top{}_{}.npy'.format(file_field, char, topk, source_type)) 
        np.save(save_path, memory)
    
    # fusion
    if fusion_type == 'add':
        fused_memory = np.stack(all_memory).mean(0)
    else:
        fused_memory = np.concatenate(all_memory, axis=-1)
    
    np.save(fused_memory_path, fused_memory)
    

def run_visual_memory(
        args: object, 
        model: Union[ModelEnsemble, Model], 
        device: torch.device,
    ) -> None:
    
    model.eval()
    model.to(device)
    
    root, wid2relevant, file_field = get_preliminary(args, model, device)
    
    if args.visual_memory_example_word:
        print('- Showing {} most relevant visual content for the specified word `{}`'.format(
            args.visual_memory_example_topk, args.visual_memory_example_word))
        
        path_to_load_videos = args.path_to_load_videos
        if not path_to_load_videos:
            path_to_load_videos = os.path.join(Constants.base_data_path, model.hparams.opt['dataset'], 'all_videos')
            
        print('- The path to load video files is {}'.format(path_to_load_videos))
        if not os.path.exists(path_to_load_videos):
            raise FileNotFoundError('Please pass the argument `--path_to_load_videos $path` to specify the path to load video files')

        os.makedirs(args.visual_memory_example_save_path, exist_ok=True)
        save_name = args.visual_memory_example_save_name
        if not save_name:
            save_name = '{}.png'.format(args.visual_memory_example_word)
        save_path = os.path.join(args.visual_memory_example_save_path, save_name)

        if len(wid2relevant) == 1:
            index_of_image_modality = 0
        else:
            index_of_image_modality = model.hparams.opt['modality'].lower().index('i')
        
        plot_visual_memory_example(
            wid2relevant[index_of_image_modality], 
            word=args.visual_memory_example_word,
            topk=args.visual_memory_example_topk, 
            vocab=model.get_vocab(),
            path_to_load_videos=path_to_load_videos,
            video_suffix=args.video_suffix,
            n_frames=model.hparams.opt['n_frames'],
            save_path=save_path
        )
    else:
        generate_visual_memory(
            wid2relevant,
            model=model,
            device=device,
            topk=args.visual_memory_topk,
            root=root,
            file_field=file_field,
            source_type=args.visual_memory_source_type,
            fusion_type=args.visual_memory_fusion_type,
        )

