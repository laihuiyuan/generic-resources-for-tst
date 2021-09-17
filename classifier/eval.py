# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch import cuda
from transformers import BartTokenizer

sys.path.append("")
from classifier.textcnn import TextCNN, TCNNIterator
from classifier.textcnn import filter_sizes, num_filters

device = 'cuda' if cuda.is_available() else 'cpu'


def main():
    parser = argparse.ArgumentParser('TextCNN Classifier for style evaluation')
    parser.add_argument('-order', default=0, type=str, help='the output order')
    parser.add_argument('-embed_dim', default=300, type=int, help='the embedding size')
    parser.add_argument('-dataset', default='fr', type=str, help='the name of dataset')
    parser.add_argument('-model', default='bart', type=str, help='style transfer model')
    parser.add_argument('-seed', default=42, type=int, help='pseudo random number seed')
    parser.add_argument('-batch_size', default=32, type=int, help='max sents in a batch')
    parser.add_argument("-dropout", default=0.6, type=float, help="Keep prob in dropout")

    opt = parser.parse_args()
    torch.manual_seed(opt.seed)

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    test_src, test_tgt = [], []
    # with open('data/{}/test.0'.format(opt.dataset), 'r') as f:
    with open('data/outputs/{}_{}_{}.1'.format(opt.model, opt.dataset, opt.order),'r') as f:
        for line in f.readlines():
            test_src.append(tokenizer.encode(line.strip()))
    # with open('data/{}/test.1'.format(opt.dataset), 'r') as f:
    with open('data/outputs/{}_{}_{}.0'.format(opt.model, opt.dataset, opt.order),'r') as f:
        for line in f.readlines():
            test_tgt.append(tokenizer.encode(line.strip()))
    print('[Info] {} instances from src test set'.format(len(test_src)))
    print('[Info] {} instances from tgt test set'.format(len(test_tgt)))
    test_loader = TCNNIterator(test_src, test_tgt, opt)

    loss_fn = nn.CrossEntropyLoss()
    model = TextCNN(opt.embed_dim, len(tokenizer), filter_sizes, num_filters)
    model.load_state_dict(torch.load('checkpoints/textcnn_{}.chkpt'.format(opt.dataset)))
    model.to(device).eval()

    total_num = 0.
    corre_num = 0.
    loss_list = []
    with torch.no_grad():
        for i,batch in enumerate(test_loader):
            x_batch, y_batch = map(lambda x: x.to(device), batch)
            logits = model(x_batch)
            y_hat = logits.argmax(dim=-1)
            same = [float(p == q) for p, q in zip(y_batch, y_hat)]
            corre_num += sum(same)
            total_num += len(y_batch)
            loss_list.append(loss_fn(logits, y_batch).item())

    print('[Info] Test: {}'.format('acc {:.4f}% | loss {:.4f}').format(
          corre_num/total_num*100, np.mean(loss_list)))


if __name__ == '__main__':
    main()
