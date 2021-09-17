# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import argparse
import numpy as np

from bleurt import score
import tensorflow as tf

import torch
import torch.nn as nn
from torch import cuda
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import BartTokenizer
from transformers import GPT2LMHeadModel

from model import BartModel
from model import BartForMaskedLM
from utils.dataset import BartIterator
from classifier.textcnn import TextCNN
from utils.optim import ScheduledOptim
from utils.helper import cal_bleu_loss, cal_bleurt_loss
from utils.helper import optimize, sample_3d, cal_sc_loss
from classifier.textcnn import num_filters, filter_sizes

device = 'cuda' if cuda.is_available() else 'cpu'


def evaluate(model, valid_loader, tokenizer, step):
    """
    Evaluation function for fine-tuning BART

    Args:
        model: the BART model.
        valid_loader: pytorch valid DataLoader.
        tokenizer: BART tokenizer
        step: the current training step.

    Returns:
        the average cross-entropy loss
    """
    
    loss_ce=[]
    with torch.no_grad():
        model.eval()
        for batch in valid_loader:
            src, tgt = map(lambda x: x.to(device), batch)
            mask = src.ne(tokenizer.pad_token_id).long()
            loss = model(src, attention_mask=mask,lm_labels=tgt)[0]
            loss_ce.append(loss.item())
        model.train()
    print('[Info] valid {:05d} | loss_cen {:.4f}'.format(step, np.mean(loss_ce)))

    return np.mean(loss_ce)


def main():
    parser = argparse.ArgumentParser('Supervised training with sentence-pair')
    parser.add_argument('-seed', default=42, type=int, help='the random seed')
    parser.add_argument('-lr', default=1e-5, type=float, help='the learning rate')
    parser.add_argument('-order', default=0, type=str, help='the order of training')
    parser.add_argument('-style', default=0, type=int, help='transfer inf. to for.')
    parser.add_argument('-model', default='bart', type=str, help='the name of model')
    parser.add_argument('-dataset', default='fr', type=str, help='the name of dataset')
    parser.add_argument('-max_len', default=20, type=int, help='max length of decoding')
    parser.add_argument('-shuffle', default=False, type=bool, help='shuffle train data')
    parser.add_argument('-steps', default=20001, type=int, help='force stop at x steps')
    parser.add_argument('-batch_size', default=32, type=int, help='the size in a batch')
    parser.add_argument('-patience', default=3, type=int, help='early stopping fine-tune')
    parser.add_argument('-eval_step', default=500, type=int, help='evaluate every x step')
    parser.add_argument('-log_step', default=100, type=int, help='print log every x step')

    opt = parser.parse_args()
    print('[Info]', opt)
    torch.manual_seed(opt.seed)

    base = BartModel.from_pretrained("facebook/bart-base")
    model = BartForMaskedLM.from_pretrained('facebook/bart-base',
                                            config=base.config)
    # model.load_state_dict(torch.load('/data/p300838/models/{}_{}_{}.chkpt'.format(
    #                       opt.model, 'pp', 1)))
    model.to(device).train()
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    eos_token_id = tokenizer.eos_token_id

    # style classifier
    cls = TextCNN(300, len(tokenizer), filter_sizes, num_filters)
    cls.load_state_dict(torch.load('checkpoints/textcnn_{}.chkpt'.format(
                  opt.dataset)))
    cls.to(device).eval()
     
    # load BLEURT model
    # config = tf.compat.v1.ConfigProto()
    # config.gpu_options.allow_growth = True
    # tf.compat.v1.Session(config=config)
    # bleur_dir = 'checkpoints/bleurt-base-128'
    # bleurt = score.BleurtScorer(bleurt_dir)

    # load data for training
    data_iter = BartIterator(tokenizer, opt)
    train_loader, valid_loader = data_iter.loader

    optimizer = ScheduledOptim(
        torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                         betas=(0.9, 0.98), eps=1e-09), opt.lr, len(train_loader))

    tab = 0
    A, B = 0, 1
    avg_loss = 1e9
    total_loss_cen = []
    total_loss_cls = []
    total_loss_bl0 = []
    total_loss_bl1 = [0]
    start = time.time()
    train_iter = iter(iter(train_loader))
        
    for step in range(1, opt.steps):

        try:
            batch = next(train_iter)
        except:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        src_seq, tgt_seq = map(lambda x: x.to(device), batch)
        mask = src_seq.ne(tokenizer.pad_token_id).long()

        # Alternate use of two different style classification based rewards
        if A==1:
            outs = model.decode(src_seq, mask, opt.max_len, False)
            loss_cls = cal_sc_loss(outs, None, cls, eos_token_id, opt.style, False)
            optimize(optimizer, loss_cls)

        else:
            outs = model(src_seq, attention_mask=mask, lm_labels=tgt_seq)
            loss_cen, logits = outs[0], outs[1]
            
            lens = src_seq.ne(tokenizer.pad_token_id).sum(-1)
            loss_cls = cal_sc_loss(logits, lens, cls, eos_token_id, opt.style)
            loss_bl0 = cal_bleu_loss(logits, src_seq, lens, eos_token_id)
            # loss_bl1 = cal_bleurt_loss(logits, src_seq, lens, tokenizer, bleurt)

            total_loss_cls.append(loss_cls.item())
            total_loss_cen.append(loss_cen.item())
            total_loss_bl0.append(loss_bl0.item())
            # total_loss_bl1.append(loss_bl1.item())

            optimize(optimizer, loss_cen+loss_cls+loss_bl0)

        if step % 10 == 0:
            A, B = B, A

        if step % opt.log_step == 0:
            lr = optimizer._optimizer.param_groups[0]['lr']
            print('[Info] steps {:05d} | loss_cen {:.4f} | loss_cls {:.4f} | '
                  'loss_bl0 {:.4f} | loss_bl1 {:.4f} | lr {:.6f} | second {:.2f}'.format(
                step, np.mean(total_loss_cen), np.mean(total_loss_cls),
                np.mean(total_loss_bl0), np.mean(total_loss_bl1),lr, time.time() - start))
            total_loss_cen = []
            total_loss_cls = []
            total_loss_bl0 = []
            total_loss_bl1 = [0]
            start = time.time()

        if step % opt.eval_step == 0:
            eval_loss = evaluate(model, valid_loader, tokenizer, step)
            if avg_loss >= eval_loss:
                torch.save(model.state_dict(), 'checkpoints/{}_{}_{}_{}.chkpt'.format(
                           opt.model, opt.dataset, opt.order, opt.style))
                print('[Info] The checkpoint file has been updated.')
                avg_loss = eval_loss
                tab = 0
            else:
                tab += 1
            if tab == opt.patience:
                exit()

if __name__ == "__main__":
    main()

