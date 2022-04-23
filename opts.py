import argparse
from config import Constants
import os
import pickle
import yaml


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='MSRVTT', help='MSRVTT | Youtube2Text')
    parser.add_argument('-m', '--modality', type=str, default='mi')
    parser.add_argument('-df', '--default', default=False, action='store_true')
    parser.add_argument('-scope', '--scope', type=str, default='')
    parser.add_argument('-method', '--method', type=str, default='', help='method to use, defined in config/methods.yaml')
    parser.add_argument('-task', '--task', type=str, default='', help='task to use, defined in config/tasks.yaml')
    parser.add_argument('-feats', '--feats', type=str, default='', help='features to use, defined in config/feats.yaml')

    parser.add_argument('--encoder', type=str, default='Encoder_preLN', help='which ')
    parser.add_argument('--decoder', type=str, default='BertDecoder', help='specify the decoder if we want')
    parser.add_argument('--cls_head', type=str, default='NaiveHead')
    parser.add_argument('--decoding_type', type=str, default='ARFormer', help='ARFormer | NARFormer')
    parser.add_argument('--fusion', type=str, default='temporal_concat', help='temporal_concat | addition')

    parser.add_argument('--attr_has_ln', default=False, action='store_true')

    model = parser.add_argument_group(title='Common Model Settings')
    model.add_argument('--dim_hidden', type=int, default=512, help='size of the hidden layer')
    model.add_argument('--encoder_dropout', type=float, default=0.5, help='strength of dropout in the encoder')
    model.add_argument('--hidden_dropout_prob', type=float, default=0.5, help='strength of dropout in the decoder')
    model.add_argument('-wc', '--with_category', default=False, action='store_true',
                        help='specified for the MSRVTT dataset, use category tags or not')
    model.add_argument('--num_category', type=int, default=20)
    model.add_argument('--textual_memory_len', type=int, default=8, help='works if the decoder has the textual memory')
    model.add_argument('--init_memory_weights', default=False, action='store_true')
    model.add_argument('--encoder_act', default='relu', type=str)

    model.add_argument('--pretrained_embs_path', type=str, default='', 
                        help='path to load pretrained word embs, which will be fixed if specified; '
                        'default to empty string, i.e., the model uses trainable word embs of dimension `dim_hidden`')
    model.add_argument('--load_model_weights_from', type=str, default='',
                        help='if specified, initializing the model with specific checkpoint file (not strict)')
    model.add_argument('--freeze_parameters_except', type=str, default=[], nargs='+',
                        help='when `load_model_weights_from` is True, specified parameters will not be frozen during training; '
                        'if not specified (default), all paramters are trainable')

    model_tf = parser.add_argument_group(title='Transformer Model Settings')
    model_tf.add_argument('--trainable_pe', default=False, action='store_true', help='use fixed (default) or trainable positional embs')
    model_tf.add_argument('--mha_exclude_bias', default=False, action='store_true')
    model_tf.add_argument('--num_hidden_layers_encoder', type=int, default=1)
    model_tf.add_argument('--num_hidden_layers_decoder', type=int, default=1)
    model_tf.add_argument('--num_attention_heads', type=int, default=8)
    model_tf.add_argument('--intermediate_size', type=int, default=2048)
    model_tf.add_argument('--hidden_act', type=str, default='relu')
    model_tf.add_argument('--attention_probs_dropout_prob', type=float, default=0.0)
    model_tf.add_argument('--with_layernorm', default=False, action='store_true')
    model_tf.add_argument('--layer_norm_eps', type=float, default=1e-12)
    model_tf.add_argument('--watch', type=int, default=0)
    model_tf.add_argument('--pos_attention', default=False, action='store_true')
    model_tf.add_argument('--enhance_input', type=int, default=2, 
                        help='for NA decoding, 0: without R | 1: re-sampling(R)) | 2: meanpooling(R), default to 2',
                        choices=[0, 1, 2])

    model_rnn = parser.add_argument_group(title='RNN Model Settings')
    model_rnn.add_argument('--rnn_type', default='lstm', type=str, help='the basic unit of RNN based decoders', choices=['lstm', 'gru'])
    model_rnn.add_argument('--forget_bias', default=0.6, type=float, help='the bias of the forget gate of LSTM')
    model_rnn.add_argument('--with_multileval_attention', default=False, action='store_true', 
                        help='also known as multimodal attention or attentional attention')
    model_rnn.add_argument('--feats_share_weights', default=False, action='store_true', 
                        help='in temporal attention, share the weights of different features or not')


    training = parser.add_argument_group(title='Common Training Settings')
    training.add_argument('-gpus', '--gpus', default=1, type=int, 
                        help='the number of gpus to use, only support 0 (cpu) and 1 now', choices=[0, 1])
    training.add_argument('-seed', '--seed', default=0, type=int, help='for reproducibility')
    training.add_argument('-e', '--epochs', type=int, default=50, help='number of epochs')
    training.add_argument('-b', '--batch_size', type=int, default=64, help='minibatch size')
    training.add_argument('--max_steps', type=int ,default=None, 
                        help='training will stop if `max_steps` or `epochs` have reached (earliest), default to None')

    training_rnn = parser.add_argument_group(title='RNN Training Settings')
    # schedule sampling: https://arxiv.org/pdf/1506.03099.pdf
    training_rnn.add_argument('--schedule_sampling_max_prob', default=0.25, type=float, help='maximum schedule sampling prob')
    training_rnn.add_argument('--schedule_sampling_saturation_epoch', default=25, type=int, help='which epoch to reach the peak value')

    training_na = parser.add_argument_group(title='Non-Autoregressive Model Training Settings')
    training_na.add_argument('--teacher_path', type=str, default='', help='path for the AR-B model')
    training_na.add_argument('--beta', nargs='+', type=float, default=[0, 1],
                        help='len=2, [lowest masking ratio, highest masking ratio]')
    training_na.add_argument('--visual_word_generation', default=False, action='store_true')
    training_na.add_argument('--demand', nargs='+', type=str, default=['VERB', 'NOUN'], 
                        help='pos_tag we want to focus on when training with visual word generation')
    training_na.add_argument('-nvw', '--nv_weights', nargs='+', type=float, default=[0.8, 1.0],
                        help='len=2, weights of visual word generation and caption generation (or mlm)')
    training_na.add_argument('--load_teacher_weights', default=False, action='store_true',
                        help='specified for NA-based models, initialize randomly or inherit from the teacher (AR-B)')


    optim_scheduler = parser.add_argument_group(title='Optimizer & LR Scheduler Settings')
    optim_scheduler.add_argument('--learning_rate', default=5e-4, type=float, help='the initial larning rate')
    optim_scheduler.add_argument('--weight_decay', type=float, default=5e-4, help='strength of weight regularization')
    optim_scheduler.add_argument('--filter_weight_decay', default=False, action='store_true', 
                        help='do not apply weight_decay on specific parametes')
    optim_scheduler.add_argument('--filter_biases', default=False, action='store_true',
                        help='if True, not applying weight decay on biases')

    optim_scheduler.add_argument('--gradient_clip_val', default=0.0, type=float, help='gradient clipping value')
    optim_scheduler.add_argument('--lr_scheduler_type', default='linear', type=str, 
                        help='`linear` (default): StepLR | `warmup`: StepLR with Warm Up, otherwise: ReduceLROnPlateau')
    # if `lr_scheduler_type` == 'linear'
    optim_scheduler.add_argument('--lr_decay', default=0.9, type=float, help='the decay rate of learning rate per epoch')
    optim_scheduler.add_argument('--lr_step_size', default=1, type=int, help='period of learning rate decay')
    # otherwise
    optim_scheduler.add_argument('--lr_monitor_mode', default='max', type=str, 
                        help='max (default): higher the metric, better the performance | min: just the opposite',
                        choices=['min', 'max'])
    optim_scheduler.add_argument('--lr_monitor_metric', default='CIDEr', type=str, help='specify the metric for lr adjustment')
    optim_scheduler.add_argument('--lr_monitor_patience', default=1, type=int, help='number of epochs with no improvement after which lr will be reduced')
    optim_scheduler.add_argument('--min_lr', default=1e-6, type=float, help='the minimum learning rate')
    optim_scheduler.add_argument('--warm_up_epoch', default=5, type=int, help='used when lr_scheduler_type is set to warmup')


    evaluation = parser.add_argument_group(title='Common Evaluation Settings')
    evaluation.add_argument('--check_val_every_n_epoch', type=int, default=1, 
                        help='check on the validation set every n train epochs, default to 1')
    evaluation.add_argument('--metric_sum', nargs='+', type=int, default=[1, 1, 1, 1],
                        help='which metrics to calculate `Sum`, default to [1, 1, 1, 1], '
                        'i.e., `Sum` = `Bleu_4` + `METEOR` + `ROUGE_L` + `CIDEr`')
    evaluation.add_argument('--save_csv', default=False, action='store_true',
                        help='save test results to csv file')

    evaluation_ar = parser.add_argument_group(title='Autoregressive Model Evaluation Settings')
    evaluation_ar.add_argument('-bs', '--beam_size', type=int, default=5,
                        help='specified for AR decoding, the number of candidates')
    evaluation_ar.add_argument('-ba', '--beam_alpha', type=float, default=1.0,
                        help='the preference of the model towards the average sentence length, '
                        'the larger `beam_alpha` is, the longer is the average sentence length')
    
    evaluation_na = parser.add_argument_group(title='Non-Autoregressive Model Evaluation Settings')
    evaluation_na.add_argument('--paradigm', type=str, default='mp', 
                        help='mp: MaskPredict | l2r: Left2Right | ef: EasyFirst')
    evaluation_na.add_argument('-lbs', '--length_beam_size', type=int, default=6,
                        help='specified for NA decoding, the number of length candidates')
    evaluation_na.add_argument('--iterations', type=int, default=5,
                        help='the number of iterations for the MP algorithm')
    evaluation_na.add_argument('--q', type=int, default=1,
                        help='the number of tokens to update for L2R & EF algorithms')
    evaluation_na.add_argument('--q_iterations', type=int, default=1,
                        help='the number of iterations for L2R & EF algorithms')
    evaluation_na.add_argument('--use_ct', default=False, action='store_true', 
                        help='use coarse-grained templates or not, only for methods with visual word generation')


    checkpoint = parser.add_argument_group(title='Checkpoint Settings')
    checkpoint.add_argument('--monitor_metric', type=str, default='Sum',
                            help='which metric to monitor for checkpoint saving: Bleu_4 | METEOR | ROUGE_L | CIDEr | Sum (default)',
                            choices=['Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'Sum'])
    checkpoint.add_argument('--monitor_mode', type=str, default='max',
                            help='max: the higher the `monitor_metric` the better the performance | min: just the opposite',
                            choices=['min', 'max'])    
    checkpoint.add_argument('--save_topk_models', type=int, default=1,
                            help='checkpoints with top-k performance will be saved, default to 1')
    checkpoint.add_argument('--save_every_10k', default=False, action='store_true',
                            help='save weight every 10k step')
    checkpoint.add_argument('--save_at_step', default=-1, type=int,
                            help='save checkpoint at this step, valid num should > 0')


    dataloader = parser.add_argument_group(title='Dataloader Settings')
    dataloader.add_argument('--max_len', type=int, default=30, help='max length of captions')
    dataloader.add_argument('--n_frames', type=int, default=8, help='the number of frames to represent a whole video')
    dataloader.add_argument('--n_caps_per_video', type=int, default=0, 
                            help='the number of captions per video to constitute the training set, '
                            'default to 0 (i.e., loading all captions for a video)')
    dataloader.add_argument('--random_type', type=str, default='segment_random', 
                            help='sampling strategy during training: segment_random (default) | all_random | equally_sampling',
                            choices=['segment_random', 'all_random', 'equally_sampling'])
    dataloader.add_argument('--load_feats_type', type=int, default=1, 
                            help='load feats from the same frame_ids (0) '
                            'or different frame_ids (1, default), '
                            'or directly load all feats without sampling (2)', 
                            choices=[0, 1, 2])
    dataloader.add_argument('--num_workers', type=int, default=1, help='num_workers for dataloader, speed up training')

    # modality information
    dataloader.add_argument('--dim_a', type=int, default=1, help='feature dimension of the audio modality')
    dataloader.add_argument('--dim_m', type=int, default=2048, help='feature dimension of the motion modality')
    dataloader.add_argument('--dim_i', type=int, default=2048, help='feature dimension of the image modality')
    dataloader.add_argument('--dim_o', type=int, default=1, help='feature dimension of the object modality')
    dataloader.add_argument('--dim_t', type=int, default=1)
    dataloader.add_argument('--feats_a_name', nargs='+', type=str, default=[])
    dataloader.add_argument('--feats_m_name', nargs='+', type=str, default=['motion_resnext101_kinetics_duration16_overlap8.hdf5'])
    dataloader.add_argument('--feats_i_name', nargs='+', type=str, default=['image_resnet101_imagenet_fps_max60.hdf5'])
    dataloader.add_argument('--feats_o_name', nargs='+', type=str, default=[])
    dataloader.add_argument('--feats_t_name', nargs='+', type=str, default=[])
    # corpus information
    dataloader.add_argument('--info_corpus_name', type=str, default='info_corpus.pkl')
    dataloader.add_argument('--reference_name', type=str, default='refs.pkl')


    multitask = parser.add_argument_group(title='Multi-Task Settings')
    multitask.add_argument('--crit', nargs='+', type=str, default=['lang'], 
                            help='which training objectives to use')
    multitask.add_argument('--crit_name', nargs='+', type=str, default=['Cap Loss'], 
                            help='names to log')
    multitask.add_argument('--crit_scale', nargs='+', type=float, default=[1.0], 
                            help='scales to weight the losses of training objectives')
    multitask.add_argument('--attention_weight_loss_type', default="l2_sum",
                            help='l1 or l2, sum or topk')
    multitask.add_argument('--attention_weight_loss_topk_k', default=2, type=int,
                            help='up to memory length, at least 1')
    multitask.add_argument('--label_smoothing', default=0., type=float,
                           help='label smoothing alpha, default: 0.0, no smoothing at all')
    multitask.add_argument('--with_temperature', default=1., type=float,
                           help='temperature, see https://arxiv.org/abs/1904.09751')

    # =====================   Sample training data   =====================
    # sample_method: linear  # sampler method, linear | root | full | metric
    # sample_minimum_percentage: 0.01
    # sample_fully_t: 20  # epoch when all data is used
    # sample_root_p: 2  # parameter used in method root
    # difficulty_type: rarity  # rarity | length | rarity_div_length | *_hard
    # try_harder: add (maybe repeated) data whose difficulty lies in 50% - 75%
    parser.add_argument("--sample_method", default="full", type=str)
    parser.add_argument("--sample_minimum_percentage", default=0.01, type=float)
    parser.add_argument("--sample_fully_t", default=30, type=int)
    parser.add_argument("--sample_root_p", default=2, type=int)
    parser.add_argument("--difficulty_type", default="rarity", type=str)
    parser.add_argument("--try_harder", action="store_true")
    parser.add_argument("--slide", action="store_true")
    parser.add_argument("--slide_type", default="expand", type=str, choices=["expand", "shift", "shiftAndExpand", "middleExpand"])
    parser.add_argument("--start_mid", default=0.0, type=float)
    parser.add_argument("--drop_first", default=0.0, type=float)
    parser.add_argument("--drop_last", default=0.0, type=float)
    parser.add_argument("--video_difficulty_path", default="vid_to_difficulty.pkl", type=str)
    parser.add_argument("--video_difficulty_weight", default=0.2, type=float)

    # =====================   Paraphrase  Augmentation   =====================
    dataloader.add_argument("--use_paraphrase", action="store_true")
    dataloader.add_argument("--paraphrase_sufix", type=str)
    dataloader.add_argument("--caption_count_file", type=str, default="/home/lishanhao/VC_data/MSVD/vid2captioncount.p")
    dataloader.add_argument("--start_after_epoch", default=0, type=int)
    dataloader.add_argument("--restore_after_epoch", default=100, type=int)
    dataloader.add_argument("--save_after_epoch", default=0, type=int)


    args = parser.parse_args()

    check_dataset(args)
    check_method(args)
    check_valid(args)
    return args


def check_valid(args):
    assert len(args.metric_sum) == 4, "[`Bleu_4', `METEOR`, `ROUGE_L`, `CIDEr`]"
    if not args.task:
        assert args.scope, "Please add the argument \'--scope $folder_name_to_save_models\'"

def check_dataset(args):
    if args.dataset.lower() == 'youtube2text':
        args.dataset = 'MSVD'
    
    assert args.dataset in ['MSVD', 'MSRVTT'], \
        "We now only support MSVD (Youtube2Text) and MSRVTT datasets."

    if args.default:
        if args.dataset == 'MSVD':
            args.beta = [0, 1]
            args.max_len = 20
            args.with_category = False
        elif args.dataset == 'MSRVTT':
            args.beta = [0.35, 0.9]
            args.max_len = 30
            args.with_category = True
    
    if args.dataset == 'MSVD':
        assert not args.with_category, \
            "Category information is not available in the Youtube2Text (MSVD) dataset"


def check_method(args):
    if args.method:
        methods = yaml.full_load(open('./config/methods.yaml'))
        assert args.method in methods.keys(), \
            "The method {} can not be found in ./config/methods.yaml".format(args.method)
        for k, v in methods[args.method].items():
            setattr(args, k, v)
    
    if args.decoding_type == 'NARFormer':
        args.crit = ['lang', 'length']
        args.crit_name = ['Cap Loss', 'Length Loss']
        args.crit_scale = [1.0, 1.0]

    if args.default:
        if args.decoding_type == 'NARFormer':
            if args.visual_word_generation:
                args.use_ct = True
                args.nv_weights = [0.8, 1.0]
            args.enhance_input = 2
            args.length_beam_size = int(6)
            args.iterations = int(5)
            args.beam_alpha = 1.35 if args.dataset == 'MSRVTT' else 1.0
            args.algorithm_print_sent = True
            args.teacher_path = os.path.join(
                Constants.base_checkpoint_path,
                args.dataset,
                'ARB',
                args.scope,
                'best.pth.tar'
            )
            assert os.path.exists(args.teacher_path)
            args.load_teacher_weights = True
            args.with_teacher = True
        else:
            args.beam_size = int(5.0)
            args.beam_alpha = 1.0
    
    if args.task:
        all_tasks = yaml.full_load(open('./config/tasks.yaml'))
        assert args.task in all_tasks.keys(), \
            "The task {} can not be found in ./config/tasks.yaml".format(args.task)

        specified_task = all_tasks[args.task]
        prefix_scope = args.task
        for k, v in specified_task.items():
            if k == 'scope_name_format':
                # override `prefix_scope`
                f, names = v
                prefix_scope = f.format(*[getattr(args, name) for name in names])
            elif 'name' in k and hasattr(args, k.replace('name', 'path')):
                arg_path = k.replace('name', 'path')
                if not getattr(args, arg_path, ''):
                    setattr(args, arg_path, os.path.join(Constants.base_data_path, args.dataset, v))
            else:
                setattr(args, k, v)
        
        args.scope = prefix_scope + '_' + args.scope if args.scope else prefix_scope

    if args.feats:
        all_feats = yaml.full_load(open('./config/feats.yaml'))
        assert args.feats in all_feats.keys()
        for k, v in all_feats[args.feats].items():
            setattr(args, k, v)
    
    args.video_difficulty_path = os.path.join(Constants.base_data_path, args.dataset, args.video_difficulty_path)


def get_dir(args, key, mid_path='', value=None):
    # for testing in Macbook
    base_path = Constants.base_data_path

    if value is None:
        value = getattr(args, key, '')
    
    if not value:
        return ''

    if isinstance(value, list):
        return [get_dir(args, key, mid_path, value=v) for v in value]
    else:
        return os.path.join(base_path, args.dataset, mid_path, value)


def where_to_save_model(args):
    return os.path.join(
        Constants.base_checkpoint_path,
        args.dataset,
        args.method,
        args.scope
    )


def get_opt():
    args = parse_opt()

    # log files and the best model will be saved at 'checkpoint_path'
    args.checkpoint_path = where_to_save_model(args)
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    # hijack infor_corpus and reference
    if args.use_paraphrase:
        if "_" not in args.paraphrase_sufix:
            args.paraphrase_sufix = "_" + args.paraphrase_sufix
        info = args.info_corpus_name.rsplit('.', 1)
        args.info_corpus_name = (args.paraphrase_sufix + ".").join(info)
        ref = args.reference_name.rsplit('.', 1)
        args.reference_name = (args.paraphrase_sufix + ".").join(ref)

    # get full paths to load features / corpora
    for key in ['feats_a_name', 'feats_m_name', 'feats_i_name', 'feats_o_name', 'feats_t_name'] \
        + ['reference_name', 'info_corpus_name']:
        setattr(args, key[:-5], get_dir(args, key, 'feats' if 'feats' in key else ''))
        print(getattr(args, key[:-5], 'None'))
        delattr(args, key)
    
    print(vars(args))
    # the assignment of 'vocab_size' should be done before defining the model
    args.vocab_size = len(pickle.load(open(args.info_corpus, 'rb'))['info']['itow'].keys())
    opt = vars(args)
    return opt
