# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import argparse

import torch
from torch import cuda
from transformers import BartTokenizer

from model import BartModel
from model import BartForMaskedLM

device = 'cuda' if cuda.is_available() else 'cpu'


def main():
    parser = argparse.ArgumentParser('generate text in target style')
    parser.add_argument('-order', default=0, type=str, help='theorder')
    parser.add_argument('-bs', default=128, type=int, help='the batch size')
    parser.add_argument('-nb', default=5, type=int, help='beam search num')
    parser.add_argument('-model', default='bart', type=str, help='model name')
    parser.add_argument('-seed', default=42, type=int, help='the random seed')
    parser.add_argument('-length', default=35, type=int, help='the max length')
    parser.add_argument('-style', default=0, type=int, help='from inf. to for.')
    parser.add_argument('-dataset', default='ye', type=str, help='dataset name')

    opt = parser.parse_args()
    torch.manual_seed(opt.seed)

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    model = BartModel.from_pretrained('facebook/bart-base')
    model.config.output_past = True
    model = BartForMaskedLM.from_pretrained("facebook/bart-base",
                                            config=model.config)
    model_dir = 'checkpoints/{}_{}_{}_{}.chkpt'.format(
                opt.model, opt.dataset, opt.order, opt.style)
    model.load_state_dict(torch.load(model_dir))
    model.to(device).eval()

    src_seq = []
    with open('data/{}/test.{}'.format(opt.dataset, opt.style)) as fin:
    # with open('data/{}/train.{}'.format(opt.dataset, opt.style)) as fin:
        for line in fin.readlines():
            src_seq.append(line.strip().lower())

    start = time.time()
    with open('./data/outputs/{}_{}_{}.{}'.format(
            opt.model, opt.dataset, opt.order, opt.style), 'w') as fout:
        for idx in range(0, len(src_seq), opt.bs):
            inp = tokenizer.batch_encode_plus(src_seq[idx: idx+opt.bs], 
                                              pad_to_max_length=True,
                                              padding=True, return_tensors='pt')
            src = inp['input_ids'].to(device)
            mask = inp['attention_mask'].to(device)
            outs = model.generate(input_ids=src,
                                  attention_mask=mask,
                                  num_beams=opt.nb,
                                  max_length=opt.length)
            for x, y in zip(outs, src_seq[idx:idx+opt.bs]):
                text = tokenizer.decode(x.tolist(), skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)
                if len(text.strip())==0:
                    text = y
                fout.write(text.strip() + '\n')


if __name__ == "__main__":
    main()

