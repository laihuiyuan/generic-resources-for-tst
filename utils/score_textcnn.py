# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch import cuda
import torch.nn.functional as F
from transformers import BartTokenizer

sys.path.append("")
from classifier.textcnn import TextCNN, TCNNIterator
from classifier.textcnn import filter_sizes, num_filters

device = 'cuda' if cuda.is_available() else 'cpu'


def main():
    parser = argparse.ArgumentParser('Score sentence-pairs using style classifier')
    parser.add_argument('-task', default='fr', type=str, help='the target task name')
    parser.add_argument('-src', default='', type=str, help='the path of source file')
    parser.add_argument('-tgt', default='', type=str, help='the path of target file')
    parser.add_argument('-embed_dim', default=300, type=int, help='the embedding size')
    parser.add_argument('-model', default='textcnn', type=str, help='the name of model')
    parser.add_argument('-seed', default=42, type=int, help='pseudo random number seed')
    parser.add_argument("-dropout", default=0.5, type=float, help="Keep prob in dropout")
    parser.add_argument('-batch_size', default=256, type=int, help='max sents in a batch')

    opt = parser.parse_args()
    torch.manual_seed(opt.seed)

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    src, tgt = [], []
    with open(opt.src, 'r') as f:
        for line in f.readlines():
            src.append(tokenizer.encode(line.strip()))
    with open(opt.tgt, 'r') as f:
        for line in f.readlines():
            tgt.append(tokenizer.encode(line.strip()))
    print('[Info] {} instances from src set'.format(len(src)))
    print('[Info] {} instances from tgt set'.format(len(tgt)))
    test_loader = TCNNIterator(src, tgt, opt, False)

    model = TextCNN(opt.embed_dim, len(tokenizer), filter_sizes, num_filters)
    model.load_state_dict(torch.load('checkpoints/textcnn_{}.chkpt'.format(opt.task)))
    model.to(device).eval()

    num = 0
    f0 = open(opt.src+'.{}.score'.format(opt.task), 'w')
    f1 = open(opt.tgt+'.{}.score'.format(opt.task), 'w')
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            x, y = map(lambda x: x.to(device), batch)
            logits = model(x)
            logits = F.softmax(logits, dim=-1)
            for score, line in zip(logits.cpu().tolist(), x):
                text = tokenizer.decode(line, skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)
                score = [str(round(s,5)) for s in score]
                if num < len(src):
                    f0.write(text.strip() + '\t' + '\t'.join(score) + '\n')
                    num += 1
                else:
                    f1.write(text.strip() + '\t' + '\t'.join(score) + '\n')

if __name__ == '__main__':
    main()
