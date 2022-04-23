import math
import pickle
from collections import defaultdict, Counter

import torch
import numpy as np
import random
import os

from functools import reduce

from torch.utils.data.sampler import SubsetRandomSampler

from config import Constants
import pandas
import json
from typing import Union, List


def get_shape_and_device(tensor):
    if isinstance(tensor, list) or isinstance(tensor, tuple):
        return get_shape_and_device(tensor[0])
    return tensor.shape, tensor.device


def to_device(
        tensor: Union[torch.Tensor, List[torch.Tensor]], 
        device: torch.device
    ) -> Union[torch.Tensor, List[torch.Tensor]]:

    if isinstance(tensor, list):
        return [to_device(item, device) for item in tensor]
    return tensor.to(device)


def to_sentence(hyp, vocab, break_words=[Constants.EOS, Constants.PAD], skip_words=[]):
    sent = []
    for word_id in hyp:
        if word_id in skip_words:
            continue
        if word_id in break_words:
            break
        word = vocab[word_id]
        sent.append(word)
    return ' '.join(sent)


def get_dict_mapping(opt, teacher_opt):
    if teacher_opt is None:
        return {}
    if teacher_opt['vocab_size'] == opt['vocab_size']:
        return {}

    info = json.load(open(opt["info_json"]))
    vocab = info['ix_to_word']

    teacher_info = json.load(open(teacher_opt["info_json"]))
    teacher_vocab = teacher_info['ix_to_word']
    teacher_w2ix = teacher_info['word_to_ix']
    if vocab == teacher_vocab:
        return {}

    dict_mapping = {}
    for k, v in vocab.items():
        dict_mapping[int(k)] = int(teacher_w2ix[v])
    return dict_mapping


def remove_repeat_n_grame(sent, n):
    length = len(sent)
    rec = {}
    result_sent = []
    for i in range(length-n+1):
        key = ' '.join(sent[i:i+n])
        if key in rec.keys():
            dis = i - rec[key] - n
            if dis in [0,1]:
                result_sent += sent[:i-dis]
                if i+n <length:
                    result_sent += sent[i+n:]
                return result_sent, False
        else:
            rec[key] = i
    return sent, True


def duplicate(sent):
    sent = sent.split(' ')
    res = {}
    for i in range(4, 0, -1):
        jud = False
        while not jud:
            sent, jud = remove_repeat_n_grame(sent, i)
            if not jud:
                res[i] = res.get(i, 0) + 1
            else:
                break
    res_str = []
    for i in range(1, 5):
        res_str.append('%d-gram: %d' % (i, res.get(i, 0)))
    return ' '.join(sent), '\t'.join(res_str)


def cal_gt_n_gram(data, vocab, splits, n=1):
    gram_count = {}
    gt_sents = {}
    for i in splits['train']:
        k = 'video%d'% int(i)
        caps = data[k]
        for tmp in caps:
            cap = [vocab[wid] for wid in tmp[1:-1]]
            gt_sents[' '.join(cap)] = gt_sents.get(' '.join(cap), 0) + 1
            for j in range(len(cap)-n+1):
                key = ' '.join(cap[j:j+n])
                gram_count[key] = gram_count.get(key, 0) + 1
    return gram_count, gt_sents


def cal_n_gram(data, n=1):
    gram_count = {}
    sents = {}
    ave_length, count = 0, 0
    for k in data.keys():
        for i in range(len(data[k])):
            sents[data[k][i]['caption']] = sents.get(data[k][i]['caption'], 0) + 1
            cap = data[k][i]['caption'].split(' ')
            ave_length += len(cap)
            count += 1
            for j in range(len(cap)-n+1):
                key = ' '.join(cap[j:j+n])
                gram_count[key] = gram_count.get(key, 0) + 1
    return gram_count, sents, ave_length/count, count


def analyze_length_novel_unique(gt_data, data, vocab, splits, n=1, calculate_novel=True):
    novel_count = 0
    hy_res, hy_sents, ave_length, hy_count = cal_n_gram(data, n)
    if calculate_novel:
        gt_res, gt_sents = cal_gt_n_gram(gt_data, vocab, splits, n)
        for k1 in hy_sents.keys():
            if k1 not in gt_sents.keys():
                novel_count += 1

    novel = novel_count / hy_count
    unique = len(hy_sents.keys()) / hy_count
    vocabulary_usage = len(hy_res.keys())

    gram4, _, _, _ = cal_n_gram(data, n=4)
    return ave_length, novel, unique, vocabulary_usage, hy_res, len(gram4)


def get_words_with_specified_tags(word_to_ix, seq, index_set, demand=['NOUN', 'VERB'], ignore_words=['is', 'are', '<mask>']):
    import nltk
    assert isinstance(index_set, set)
    res = nltk.pos_tag(seq.split(' '))
    for w, t in res:
        if Constants.pos_tag_mapping[t] in demand and w not in ignore_words:
            index_set.add(word_to_ix[w])


def enlarge(info, beam_size):
    bsz, *rest_shape = info.shape
    if len(rest_shape) == 2:
        info = info.unsqueeze(1).repeat(1, beam_size, 1, 1)
    elif len(rest_shape) == 1:
        info = info.unsqueeze(1).repeat(1, beam_size, 1)
    else:
        info = info.unsqueeze(1).repeat(1, beam_size)
    return info.contiguous().view(bsz * beam_size, *rest_shape)


def auto_enlarge(info, beam_size):
    if isinstance(info, list):
        if isinstance(info[0], tuple):
            return [
                tuple([enlarge(_, beam_size) for _ in item])
                for item in info
            ]
        else:
            return [enlarge(item, beam_size) for item in info]
    else:
        if isinstance(info, tuple):
            return tuple([enlarge(item, beam_size) for item in info])
        else:
            return enlarge(info, beam_size)


def print_information(opt, model):
    print(model)
    print('| model {}'.format(opt['method']))
    print('| num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))
    print('dataloader random type: %s' % opt.get('random_type', 'segment_random'))
    print('k best model: %d' % opt.get('k_best_model', 10))
    print('modality: %s' % opt['modality'])
    print('n frames: %d' % opt['n_frames'])
    print('max_len: %d' % opt['max_len'])
    print('vocab_size: %d' % opt['vocab_size'])
    print('seed: %d' % opt['seed'])
    print('teacher_path: %s' % opt.get('teacher_path', ""))


def filter_weight_decay(model, weight_decay=1e-5, filter_biases=False, skip_list=(), skip_substr_list=()):
    def is_substr_in(name):
        for substr in skip_substr_list:
            if substr in name:
                return True
        return False

    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if filter_biases and param.dim() == 1:
            no_decay.append(param)
        elif name in skip_list or is_substr_in(name):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def resampling(source_length, target_length):
    return [round(i * (source_length-1) / (target_length-1)) for i in range(target_length)]


def get_uniform_ids_from_k_snippets(length, k, offset=0):
    uniform_ids = []
    bound = [int(i) for i in np.linspace(0, length, k+1)]
    for i in range(k):
        idx = (bound[i] + bound[i+1]) // 2
        uniform_ids.append(idx + offset)
    return uniform_ids


def get_random_ids_from_k_snippets(length, k, offset=0):
    random_ids = []
    bound = [int(i) for i in np.linspace(0, length, k+1)]
    for i in range(k):
        idx = np.random.randint(bound[i], bound[i+1])
        random_ids.append(idx + offset)
    return random_ids


def get_random_ids_from_the_whole(length, k, offset=0):
    random_ids = random.sample([i for i in range(length)], k)
    random_ids = [i + offset for i in random_ids]
    return sorted(random_ids)


def get_uniform_items_from_k_snippets(items, k):
    uniform_ids = get_uniform_ids_from_k_snippets(len(items), k)
    return [items[idx] for idx in uniform_ids]


def get_ids_of_keyframes(total_frames_of_a_video, k, identical=True, offset=0):
    if identical:
        ''' In our implementation, we follow two steps:
            1. extract 60 features to represent a video (see the `hdf5` feature files);
            2. feed uniformly-sampled k features into the captioning model during inference.
        '''
        assert k < 60
        uniform_ids = get_uniform_ids_from_k_snippets(total_frames_of_a_video, 60) # step1
        real_ids = get_uniform_items_from_k_snippets(uniform_ids, k) # step2
    else:
        ''' the real_ids is slightly different from the one above
            e.g., with total_frames_of_a_video = 198 and k = 8,
            identical = True:  real_ids = [11, 37, 60, 87, 110, 136, 159, 186]
            identical = False: real_ids = [12, 36, 61, 86, 111, 135, 160, 185]
        '''
        real_ids = get_uniform_ids_from_k_snippets(total_frames_of_a_video, k)

    if offset:
        real_ids = [idx + offset for idx in real_ids]

    return real_ids


def save_dict_to_csv(path, file_name, dict_data):
    if ".csv" not in file_name:
        file_name = file_name + ".csv"
    csv_path = os.path.join(path, file_name)
    df_scores = pandas.DataFrame([dict_data])
    df_scores.to_csv(csv_path, index=False, mode='a', header=not os.path.exists(csv_path))


def get_idx_by_sorted_difficulty(dataset, caption_difficulty_type="rarity", category_enhance_type=None,
                                 video_difficulty_path="", video_difficulty_weight=0.3):
    """
    Note that idx within this code refers to the index of caption instance from `dataset.infoset`,
    which might be different from the caption id.
    At least one of caption or video should be valid, or the returned idx is none.
    Args:
        dataset: dataset
        caption_difficulty_type: "" | rarity | length | rarity_div_length | *_hard
        category_enhance_type: "" | oversample
        path_to_video_difficulties: a list of path to such video difficulty file
        video_difficulty_keys: depends on the file(s)
        weights: a list of float
    """

    def get_difficulty(caption, idx_to_count, total_count, difficulty_type):
        if "rarity_div_length" in difficulty_type:
            neg_log_count_percent = map(lambda x: - np.log(idx_to_count[x] / total_count), caption)
            difficulty = reduce(lambda x, y: x + y, neg_log_count_percent) / len(caption)
        elif "length" in difficulty_type:
            difficulty = len(caption)
        elif "rarity" in difficulty_type:
            neg_log_count_percent = map(lambda x: - np.log(idx_to_count[x] / total_count), caption)
            difficulty = reduce(lambda x, y: x + y, neg_log_count_percent)
        else:
            raise NotImplementedError(f"{difficulty_type} is not supported now.")

        if "hard" in difficulty_type:
            # reverse the order
            difficulty = -difficulty
        return difficulty

    # oversample categories whose amount below 0.75 * average
    if category_enhance_type == "oversample":
        # if category, add duplicate captions from rare categories, but no sorting
        idx_to_category = [x["category"] for x in dataset.infoset]
        category_to_count = Counter(idx_to_category)
        mean_count = 0.75 * sum(category_to_count.values()) / len(category_to_count.values())
        # target_count = 0.5 * max(category_to_count.values()) + 0.5 * mean_count
        target_count = mean_count
        # category_to_idx = defaultdict(list)
        # for idx, cate in enumerate(idx_to_category):
        #     category_to_idx.append(idx)
        add_to_infoset = []
        for ist in dataset.infoset:
            cate = ist["category"]
            if category_to_count[cate] < target_count:
                repeat = int(target_count / category_to_count[cate])
                repeat = max(repeat, 0)
                add_to_infoset += [ist for _ in range(repeat)]
                # category_to_count[cate] += repeat
        dataset.infoset += add_to_infoset

    #  [1:-1] : remove <bos> and <eos> to calculate difficulty of the real sentences
    captions = list(map(lambda x: x["labels"][1:-1], dataset.infoset))
    # train_data = self.corpus[0]
    word_idx_to_count = Counter()
    for caption in captions:
        word_idx_to_count.update(caption)
    total_count = sum(word_idx_to_count.values())

    idx_to_difficulty = defaultdict(float)

    # caption difficulty
    for idx, (caption) in enumerate(captions):
        difficulty = get_difficulty(caption, word_idx_to_count, total_count, caption_difficulty_type)
        idx_to_difficulty[idx] = difficulty
    # add video difficulty
    # normalize both difficulty
    if "video" in caption_difficulty_type.split("-"):
        max_caption_difficulty = max(idx_to_difficulty.values())
        
        with open(video_difficulty_path, "rb") as f:
            vid_to_difficulty = pickle.load(f)
        max_video_difficulty = max(vid_to_difficulty.values())
        for idx, difficulty in idx_to_difficulty.items():
            vid = dataset.infoset[idx]['vid']
            caption_difficulty = difficulty / max_caption_difficulty
            video_difficulty = vid_to_difficulty[vid] / max_video_difficulty
            idx_to_difficulty[idx] = (1.0 - video_difficulty_weight) * caption_difficulty + video_difficulty_weight * video_difficulty

    idx_with_difficulty = sorted(idx_to_difficulty.items(), key=lambda x: x[-1])  # sort by difficulty
    idxs = list(map(lambda x: x[0], idx_with_difficulty))

    if "video_level" in caption_difficulty_type.split("-"):
        idx_to_vid = [x["vid"] for x in dataset.infoset]
        vid_to_difficulty_with_idx = defaultdict(list)
        for idx, vid in enumerate(idx_to_vid):
            vid_to_difficulty_with_idx[vid].append([idx_to_difficulty[idx], idx])
        vid_to_sorted_difficulty_with_idx = {}
        for vid, lst in vid_to_difficulty_with_idx.items():
            vid_to_sorted_difficulty_with_idx[vid] = sorted(lst)  # intra-video sort
        vid_to_select_idx = {vid: 0 for vid in vid_to_difficulty_with_idx.keys()}
        candidate_vids = set(vid_to_difficulty_with_idx.keys())
        num_videos = len(candidate_vids)
        idxs = []
        while len(candidate_vids) > 0:
            difficulty_with_idx = []  # for each video, select one
            next_candidate_vids = set()
            for vid in candidate_vids:
                difficulty_with_idx.append(vid_to_sorted_difficulty_with_idx[vid][vid_to_select_idx[vid]])
                vid_to_select_idx[vid] += 1
                if vid_to_select_idx[vid] < len(vid_to_sorted_difficulty_with_idx[vid]):
                    next_candidate_vids.add(vid)
            candidate_vids = next_candidate_vids
            if len(difficulty_with_idx) >= 0.9 * num_videos:
                difficulty_with_idx = sorted(difficulty_with_idx)  # inter-video sort
            idxs += list(map(lambda x: x[1], difficulty_with_idx))
    return idxs


class Sampler(object):
    """
    Sample data by difficulty.
    """
    def __init__(self, idx_by_difficulty, opt):
        self.idx_by_difficulty = idx_by_difficulty
        self.max_epoch = opt["epochs"]
        self.method = opt["sample_method"]
        self.minimum_sample_percent = opt["sample_minimum_percentage"]
        assert self.minimum_sample_percent > 0
        self.root_p = opt["sample_root_p"]
        self.fully_t = opt["sample_fully_t"]
        # for `metric` method
        self.baseline_metric_sum = 1.8 if opt["dataset"] == "MSRVTT" else 2.4
        self.minimum_increase = 0.02
        self.previous_percentage = self.minimum_sample_percent - self.minimum_increase
        self.if_slide = opt["slide"]
        self.slide_type = opt["slide_type"]
        self.if_try_harder = opt["try_harder"]
        self.start_mid = opt["start_mid"]
        # drop some examples
        self.drop_first = opt["drop_first"]
        self.drop_last = opt["drop_last"]
        num_first_drop = int(len(self.idx_by_difficulty) * self.drop_first)
        num_last_drop = int(len(self.idx_by_difficulty) * self.drop_last)
        self.idx_by_difficulty = self.idx_by_difficulty[num_first_drop:]
        self.idx_by_difficulty = self.idx_by_difficulty[:-num_last_drop] if num_last_drop else self.idx_by_difficulty

    def get_percentage(self, current_epoch, current_metric_sum):
        c = self.minimum_sample_percent
        t = current_epoch
        T = self.fully_t
        p = self.root_p
        if self.method == "full" or current_epoch >= T:
            percentage = 1.0
        elif self.method == "linear":
            percentage = t * (1.0 - c) / T + c
        elif self.method == "root":
            base = t * (1.0 - math.pow(c, p)) / T + math.pow(c, p)
            percentage = math.pow(base, 1.0 / p)
        elif self.method == "metric":
            current_percentage = current_metric_sum / self.baseline_metric_sum
            percentage = max(self.minimum_increase + self.previous_percentage, current_percentage)
        elif self.method == "baby":
            # short for baby step
            total_step = 8
            step = T / total_step
            percentage = t // step / total_step
        else:
            raise NotImplementedError(f"sampler method should be linear | root | full | metric | baby, "
                                      f"now {self.method} is not supported.")
        percentage = max(percentage, self.minimum_sample_percent)  # guarantee minimum
        percentage = min(percentage, 1.)  # guarantee maximum, clip at 1

        self.previous_percentage = percentage

        return percentage

    def try_harder(self, size):
        last = len(self.idx_by_difficulty) // 4
        return random.choices(self.idx_by_difficulty[-last*2:-last], k=size)

    def get_sampler(self, current_epoch, current_metric_sum=0):
        total = len(self.idx_by_difficulty)
        percentage = self.get_percentage(current_epoch, current_metric_sum)
        start_num = 0
        num_selected = int(total * percentage)
        end_num = num_selected
        # assert num_selected > 0, f"num of sample should be above 0, now is {num_selected}."
        if self.if_slide:
            type = self.slide_type
            if type == "expand":
                # always start at 0, covers more and more
                pass
            elif type == "middleExpand":
                mid_num = int(total * self.start_mid)
                left_half_num = int(num_selected * self.start_mid)
                start_num = mid_num - left_half_num
                end_num = mid_num + num_selected - left_half_num
                if start_num < 0 or end_num > total:
                    start_num = 0
                    end_num = total
            else:
                raise NotImplementedError(f"slide type `{type}` is not supported.")
            print(f"sliding: {start_num}, {end_num}, amount:{end_num - start_num + 1}")

        idxs = self.idx_by_difficulty[start_num: end_num]

        if self.if_try_harder:
            idxs += self.try_harder(num_selected // 5)

        return SubsetRandomSampler(idxs)
