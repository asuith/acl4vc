# import os
# from pytorch_lightning import loggers
# import torch
# from torch import nn
# import torch.nn.functional as F
# from torch.nn.modules.activation import Softmax
# from torch.utils.data.sampler import BatchSampler
# from torchvision import transforms
# from torchvision.datasets import MNIST
# from torch.utils.data import DataLoader, random_split

# from pytorch_lightning import (
#     seed_everything, 
#     LightningModule, 
#     Trainer,
# )
# from pytorch_lightning.callbacks import Callback, ModelCheckpoint, LearningRateMonitor
# from pytorch_lightning.metrics import functional as FM
# from pytorch_lightning.loggers import TensorBoardLogger

# from tqdm import tqdm
# import warnings
# warnings.filterwarnings("ignore")

# from torch.optim.lr_scheduler import ReduceLROnPlateau


# class PrintCallback(Callback):
#     def on_train_start(self, trainer, pl_module):
#         tqdm.write("Training is started!")
#     def on_train_end(self, trainer, pl_module):
#         tqdm.write("Training is done!")
#     def on_train_epoch_start(self, trainer, pl_module):
#         tqdm.write("Training of epoch %d is started!" % trainer.current_epoch)
#     # def on_train_epoch_end(self, trainer, pl_module):
#     #     tqdm.write("Training of epoch %d is done! (%.3f)" % (trainer.current_epoch, pl_module.train_epoch_acc))
#     def on_eval_epoch_end(self, trainer, pl_module):
#         tqdm.write("Validation of epoch %d is done!" % (trainer.current_epoch))


# class Model(LightningModule):
#     def __init__(self):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Linear(28*28, 64),
#             nn.ReLU(),
#             nn.Linear(64, 10),
#             nn.Softmax(dim=-1)
#         )
#         self.lr = 0.01

#     def forward(self, batch, batch_idx, string='forward'):
#         # in lightning, forward defines the prediction/inference actions
#         x, y = batch
#         x = x.view(x.size(0), -1)
#         y_hat = self.model(x)
#         loss = F.cross_entropy(y_hat, y)
#         acc = FM.accuracy(y_hat, y)

#         metrics = {'val_acc': acc, 'val_loss': loss}
#         self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False)
#         return metrics

#     def training_step(self, batch, batch_idx):
#         # training_step defined the train loop.
#         # It is independent of forward
#         x, y = batch
#         x = x.view(x.size(0), -1)
#         y_hat = self.model(x)
#         loss = F.cross_entropy(y_hat, y)
#         acc = FM.accuracy(y_hat, y)
#         self.log_dict({'train_loss': loss, 'train_acc': acc}, on_step=True, on_epoch=True, prog_bar=False)
#         return {'loss': loss, 'acc': acc}

#     def validation_step(self, batch, batch_idx):
#         return self.forward(batch, batch_idx, string='validation')
    
#     def training_epoch_end(self, outputs) -> None:
#         pass
    
#     def validation_epoch_end(self, outputs) -> None:
#         pass

#     def test_epoch_end(self, outputs) -> None:
#         pass

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
#         lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, verbose=True)
#         return {
#             'optimizer': optimizer,
#             'lr_scheduler': {
#                 'scheduler': lr_scheduler,
#                 'interval': 'epoch',
#                 'frequency': 1,
#                 'monitor': 'val_acc',
#                 'strict': True
#             }
#         }

#     # def train_dataloader(self):
#     #     return DataLoader(dataset, shuffle=True, batch_size=64)

#     def get_progress_bar_dict(self):
#         # don't show the version number
#         items = super().get_progress_bar_dict()
#         items.pop("v_num", None)
#         return items
    
#     def add_model_specific_args(parent_parser):
#         parser = parent_parser.add_argument_group("model")
#         parser.add_argument('--encoder_layers', type=int, default=12)
#         parser.add_argument('--data_path', type=str, default='/some/path')
#         return parent_parser


# from argparse import ArgumentParser
# parser = ArgumentParser()
# # add PROGRAM level args
# parser.add_argument('--seed', type=int, default=0)
# parser.add_argument('--root_path', type=str, default='./experiments')
# parser.add_argument('--scope', type=str, default='test')

# # add model specific args
# parser = Model.add_model_specific_args(parser)

# # add all the available trainer options to argparse
# # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
# parser = Trainer.add_argparse_args(parser)

# args = parser.parse_args()

# seed_everything(0, workers=True)

# dirpath = 'dummy_test'

# checkpoint_callback = ModelCheckpoint(
#     monitor='val_acc',
#     mode='max',
#     save_last=True,
#     save_top_k=None,
#     dirpath=dirpath,
#     filename='sample-mnist-epoch{epoch:02d}-val_acc{val_acc:.4f}',
#     auto_insert_metric_name=False
# )
# logger = TensorBoardLogger(dirpath)

# callbacks = [PrintCallback(), LearningRateMonitor(logging_interval='epoch'), checkpoint_callback,]
# trainer = Trainer(
#     deterministic=True, 
#     weights_summary='full',
#     auto_lr_find=False, 
#     log_every_n_steps=200,
#     max_epochs=10,
#     max_steps=1e6,
#     reload_dataloaders_every_epoch=False,
#     resume_from_checkpoint=None,
#     check_val_every_n_epoch=1,
#     callbacks=callbacks,
#     logger=logger,
# )

# dataset = MNIST(os.getcwd(), download=True, train=True, transform=transforms.ToTensor())
# train_loader = DataLoader(dataset, shuffle=True, batch_size=64)
# vali_loader = DataLoader(MNIST(os.getcwd(), download=True, train=False, transform=transforms.ToTensor()), batch_size=64)

# # init model
# model = Model()
# # model = model.load_from_checkpoint('dummy_test/sample-mnist-epoch08-val_loss0.96.ckpt')
# # model.freeze()

# trainer.fit(model, train_loader, vali_loader)

# print(checkpoint_callback.best_model_path)
# print(checkpoint_callback.best_model_score)

# trainer.validate(model, vali_loader)

import torch
import torch.nn as nn

class Embeddings(nn.Module):
    def __init__(self):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding.from_pretrained(
            embeddings=torch.zeros((2, 4)),
            freeze=True,
        )

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.emb = Embeddings()
        print(self.emb.word_embeddings.weight)
        self._init_weights()  
        print(self.emb.word_embeddings.weight)   

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

m = Model()

