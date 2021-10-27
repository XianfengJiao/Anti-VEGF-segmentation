""" This main entrance of the whole project.

    Most of the code should not be changed, please directly
    add all the input arguments of your model's constructor
    and the dataset file's constructor. The MInterface and 
    DInterface can be seen as transparent to all your args.    
"""
import os
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import StratifiedKFold
import torch.utils.data as data
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import os
import scipy.io as scio
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

from model import MInterface
from dataset import DInterface
import pickle as pkl
from utils import load_model_path_by_args

def load_callbacks():
    callbacks = []
    # callbacks.append(plc.EarlyStopping(
    #     monitor='val_auc_scores',
    #     mode='max',
    #     patience=20,
    #     min_delta=0.001
    # ))

    callbacks.append(plc.ModelCheckpoint(
        monitor='val_loss',
        filename='best-{epoch:02d}-{val_auc_scores:.3f}',
        save_top_k=2,
        mode='min',
        save_last=True
    ))

    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval='epoch'))
    return callbacks

def get_new_labels(y):
    y_new = LabelEncoder().fit_transform([''.join(str(l)) for l in y])
    return y_new

def load_data_path(data_dir):
    data_paths = glob.glob(os.path.join(data_dir, "*.mat"))
    return np.array(data_paths)

def train(train_dataset, test_dataset, val_dataset, args):
    pl.seed_everything(args.seed)
    data_module = DInterface(args.num_workers, args.dataset, train_dataset, test_dataset, val_dataset, args.batch_size)
    load_path = load_model_path_by_args(args)

    if load_path is None:
        model = MInterface(**vars(args))
    else:
        model = MInterface(**vars(args))
        args.resume_from_checkpoint = load_path

    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, data_module)
    logged_metrics = trainer.logged_metrics
    return logged_metrics

def inference(data_paths, args):
    pl.seed_everything(args.seed)
    model = MInterface(**vars(args))
    model = model.load_from_checkpoint(checkpoint_path=args.checkpoint_path)
    for data_p in data_paths:
        data = scio.loadmat(data_p)
        img = data['img']
        mask_true = data['annotation']
        img = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0)
        model.eval()
        sigmoid = torch.nn.Sigmoid()
        mask_pred = sigmoid(model(img)).squeeze(0).detach().cpu().numpy()
        

        mask_merge = np.zeros(mask_pred[0].shape)
        
        for i, mm in enumerate(mask_pred):
            mm[mm <= 0.8] = 0
            mm[mm > 0.8] = i + 1
            mask_merge = np.maximum(mask_merge, mm)
        
        plt.imshow(mask_merge, interpolation='nearest')
        plt.show()

        plt.imshow(mask_true, interpolation='nearest')
        plt.show()

        plt.imshow(img.squeeze(0).squeeze(0).numpy(), interpolation='nearest', cmap='gray', vmin=0, vmax=255)
        plt.show()
        



def main(args):
    # TODO: Load data according to args
    dataset = load_data_path(args.data_dir)
    length=len(dataset)
    train_size, val_size=int(0.9*length),int(0.1*length)
    train_set, val_set=data.random_split(range(length),[train_size,val_size])

    # if args.inference:
    if True:
        inference(dataset[val_set], args)
        exit(0)

    logged_metrics = train(dataset[train_set], dataset[val_set], dataset[val_set], args)
    

if __name__ == '__main__':
    parser = ArgumentParser()
    # Basic Training Control
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--n2n', action='store_true')
    parser.add_argument('--kfold', default=0, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--save_path', default='./checkpoints/segmentation/UNet_cross_sigmoid', type=str)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)

    # LR Scheduler
    parser.add_argument('--lr_scheduler', choices=['step', 'cosine'], type=str)
    parser.add_argument('--lr_decay_steps', default=20, type=int)
    parser.add_argument('--lr_decay_rate', default=0.5, type=float)
    parser.add_argument('--lr_decay_min_lr', default=1e-5, type=float)

    # Restart Control
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--load_best', action='store_true')
    parser.add_argument('--load_dir', default=None, type=str)
    parser.add_argument('--load_ver', default=None, type=str)
    parser.add_argument('--checkpoint_path', default='/home/jxf/code/python/pytorch-lightning-med/segmentation/lightning_logs/version_7/checkpoints/epoch=99-step=2499.ckpt', type=str)
    parser.add_argument('--load_v_num', default=None, type=int)

    # Training Info
    parser.add_argument('--dataset', default='OCT_data', type=str)
    parser.add_argument('--data_dir', default='/home/jxf/code/python/pytorch-lightning-med/segmentation/dataset/data/Anti-VEGF', type=str)
    parser.add_argument('--model_name', default='UNet', type=str)
    parser.add_argument('--loss', default='cross entropy', type=str)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--no_augment', action='store_true')
    parser.add_argument('--log_dir', default='lightning_logs', type=str)
    
    # Model Hyperparameters
    parser.add_argument('--hid', default=64, type=int)
    parser.add_argument('--block_num', default=8, type=int)
    parser.add_argument('--in_channel', default=3, type=int)
    parser.add_argument('--layer_num', default=5, type=int)

    # Other
    parser.add_argument('--aug_prob', default=0.5, type=float)

    parser.add_argument_group(title="pl.Trainer args")
    parser = Trainer.add_argparse_args(
        parser
    )

    # Reset Some Default Trainer Arguments' Default Values
    parser.set_defaults(max_epochs=100)

    args = parser.parse_args()

    # List Arguments
    args.mean_sen = [0.485, 0.456, 0.406]
    args.std_sen = [0.229, 0.224, 0.225]

    main(args)
