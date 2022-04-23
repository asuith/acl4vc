'''
    Author: Bang Yang
'''
import warnings
warnings.filterwarnings('ignore')

from opts import get_opt
from models import Model
from dataloader import get_loader
from pytorch_lightning import (
    seed_everything, 
    Trainer,
)
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger


if __name__ == '__main__':
    opt = get_opt()

    seed_everything(opt['seed'], workers=True)
    
    if opt.get('load_model_weights_from', ''):
        model = Model.load_from_checkpoint(
            opt['load_model_weights_from'],
            new_opt_used_to_override=opt,
            strict=False if opt.get('visual_memory_late_fusion', False) or opt.get('visual_memory_early_fusion', False) else True,
            merge_opt=True,
        )
        names = set(opt.get('freeze_parameters_except', []))
        if len(names):
            print('- Parameter names that contain any string of {} are trainable'.format(names))
            for n, p in model.named_parameters():
                flag = sum([1 for specified_name in names if specified_name in n])
                if not flag:
                    p.requires_grad = False
    else:
        model = Model(opt)
    print(model)

    if opt['save_topk_models'] > 1:
        some_args_about_checkpoint = {
            'save_top_k': opt['save_topk_models'],
            'filename': 'E{epoch:02d}-B{Bleu_4:.3f}-M{METEOR:.3f}-R{ROUGE_L:.3f}-C{CIDEr:.3f}-Sum{Sum:.3f}',
            'auto_insert_metric_name': False,
        }
    else:
        some_args_about_checkpoint = {
            'save_top_k': opt['save_topk_models'],
            'filename': 'best'
        }

    checkpoint_callback = ModelCheckpoint(
        monitor=opt['monitor_metric'],
        mode=opt['monitor_mode'],
        save_last=True,
        dirpath=opt["checkpoint_path"],
        save_weights_only=True,
        **some_args_about_checkpoint
    )
    logger = TensorBoardLogger(opt["checkpoint_path"])

    # by defining callbacks below, The trainer will automatically log the learning rate and save models
    callbacks = [LearningRateMonitor(logging_interval='epoch'), checkpoint_callback,]
    
    # for testing in Macbook
    extra_args =  {'limit_train_batches': 3} if opt['scope'] in ['test', 'dummy'] else {}
    
    trainer = Trainer(
        deterministic=True,
        weights_summary='full',
        auto_lr_find=False, 
        log_every_n_steps=50,
        max_epochs=opt['epochs'],
        max_steps=opt['max_steps'],
        reload_dataloaders_every_epoch=True,
        resume_from_checkpoint=None,
        check_val_every_n_epoch=opt['check_val_every_n_epoch'],
        callbacks=callbacks,
        logger=logger,
        gpus=opt['gpus'],
        gradient_clip_val=opt['gradient_clip_val'],
        num_sanity_val_steps=0,
        **extra_args
    )
    
    # train_loader = get_loader(opt, 'train', print_info=False)
    vali_loader = get_loader(opt, 'validate', print_info=False)
    test_loader = get_loader(opt, 'test', print_info=False)

    trainer.fit(model, val_dataloaders=vali_loader)
    # trainer.fit(model)

    print('best_model_path:', checkpoint_callback.best_model_path)
    print('best_model_score', checkpoint_callback.best_model_score)

    model = Model.load_from_checkpoint(checkpoint_callback.best_model_path)
    trainer.test(model, test_loader)
