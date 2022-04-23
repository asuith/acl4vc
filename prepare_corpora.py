''' Handling the data io '''
import argparse
import shutil
from collections import defaultdict
from config import Constants
import wget
import os
import pickle
from misc import utils_corpora

# only words that occur more than this number of times will be put in vocab
word_count_threshold = {
    'MSRVTT': 2,
    'MSVD': 0
}


def main(args):
    func_name = 'preprocess_%s' % args.dataset
    preprocess_func = getattr(utils_corpora, func_name, None)
    if preprocess_func is None:
        raise ValueError('We can not find the function %s in misc/utils_corpora.py' % func_name)

    results = preprocess_func(args.file_path)
    split = results['split']
    raw_caps_train = results['raw_caps_train']
    raw_caps_all = results['raw_caps_all']
    references = results.get('references', None)

    vid2id = results.get('vid2id', None)
    itoc = results.get('itoc', None)
    split_category = results.get('split_category', None)
    
    # create the vocab
    vocab = utils_corpora.build_vocab(
        raw_caps_train, 
        word_count_threshold[args.dataset],
        sort_vocab=args.sort_vocab,
        attribute_first=args.attribute_first
        )
    itow, captions, itop, pos_tags = utils_corpora.get_captions_and_pos_tags(raw_caps_all, vocab)

    length_info = utils_corpora.get_length_info(captions)
    #next_info = get_next_info(captions, split)

    info = {
        'split': split,                # {'train': [0, 1, 2, ...], 'validate': [...], 'test': [...]}
        'vid2id': vid2id,
        'split_category': split_category,
        'itoc': itoc,
        'itow': itow,                       # id to word
        'itop': itop,                       # id to POS tag
        'length_info': length_info,         # id to length info
    }

    if args.sort_vocab and args.attribute_first:
        info['vid2attr'] = utils_corpora.get_attribute_label(captions)

    pickle.dump({
            'info': info,
            'captions': captions,
            'pos_tags': pos_tags,
        }, 
        open(args.corpus, 'wb')
    )
    print("corpus saved to %s" % args.corpus)

    if references is not None:
        pickle.dump(
            references,
            open(args.refs, 'wb')
        )
        print("references saved to %s" % args.refs)
    elif args.file_path != args.refs:
        shutil.copy(args.file_path, args.refs)
        print("ref copied and renamed as %s" % args.refs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='VATEX', type=str)
    parser.add_argument('-sort', '--sort_vocab', default=False, action='store_true')
    parser.add_argument('-attr', '--attribute_first', default=False, action='store_true')
    
    parser.add_argument('-pp', '--pretrained_path', default='', type=str, 
            help='path of the file that stores pretrained word embeddings (e.g., glove.840B.300d.txt); '
            'if specified, pretrained word embeddings of the given dataset will be extracted and stored.')
    parser.add_argument('-pd', '--pretrained_dim', default=300, type=int, 
            help='dimension of the pretrained word embeddings')
    parser.add_argument('-sn', '--save_name', default='embs.npy', type=str, 
            help='the filename to save pretrained word embeddings of the given datasets')

    # for json file after augmentation
    parser.add_argument('--version', default='', type=str,
                        help='the suffix added to the output files')
    parser.add_argument('--file_name', default='', type=str,
                        help='name to annotated json / pickle file (MSR-VTT / MSVD)')

    args = parser.parse_args()
    if args.dataset.lower() == 'youtube2text':
        args.dataset = 'MSVD'
    
    assert args.dataset in word_count_threshold.keys()

    args.base_pth = os.path.join(Constants.base_data_path, args.dataset)
    args.corpus = os.path.join(args.base_pth, f'info_corpus_{args.version}.pkl')
    args.refs = os.path.join(args.base_pth, f'refs_{args.version}.pkl')

    args.file_path = os.path.join(args.base_pth, args.file_name)

    if args.pretrained_path:
        #try:
        utils_corpora.prepare_pretrained_word_embeddings(args)
        #except:
        #    args.sort_vocab = True
        #    main(args)
        #    utils_corpora.prepare_pretrained_word_embeddings(args)
    else:
        main(args)

'''
python prepare_corpora.py --dataset MSRVTT --sort_vocab
python prepare_corpora.py --dataset MSRVTT --sort_vocab --attribute_first 

python prepare_corpora.py --dataset MSRVTT --sort_vocab --attribute_first \
--pretrained_path new_VC_data/glove.840B.300d.txt \
-pd 300 -sn glove_embs.npy
'''
