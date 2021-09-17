# -*- coding: utf-8 -*-

import random
import numpy as np

import torch
import torch.utils.data


class BartDataset(torch.utils.data.Dataset):
    """ Seq2Seq Dataset """

    def __init__(self, src_inst, tgt_inst):

        self.src_inst = src_inst
        self.tgt_inst = tgt_inst

    def __len__(self):
        return len(self.src_inst)

    def __getitem__(self, idx):
        return self.src_inst[idx], self.tgt_inst[idx]


class BartIterator(object):
    """ Data iterator for fine-tuning BART """

    def __init__(self, tokenizer, opt):

        self.tokenizer = tokenizer
        self.opt = opt

        self.train_src, self.train_tgt = self.read_insts('train', opt.shuffle, opt)
        self.valid_src, self.valid_tgt = self.read_insts('valid', False, opt)
        print('[Info] {} instances from train set'.format(len(self.train_src)))
        print('[Info] {} instances from valid set'.format(len(self.valid_src)))

        self.loader = self.gen_loader(self.train_src, self.train_tgt, 
                                      self.valid_src, self.valid_tgt)

    def read_insts(self, mode, shuffle, opt):
        """
        Read instances from input file
        Args:
            mode (str): 'train' or 'valid'.
            shuffle (bool): whether randomly shuffle training data.
            opt: it contains the information of transfer direction.
        Returns:
            src_seq: list of the lists of token ids for each source sentence.
            tgt_seq: list of the lists of token ids for each tgrget sentence.
        """

        src_dir = 'data/{}/{}.{}'.format(opt.dataset, mode, opt.style)
        tgt_dir = 'data/{}/{}.{}'.format(opt.dataset, mode, bool(opt.style-1).real)

        src_seq, tgt_seq = [], []
        with open(src_dir, 'r') as f1, open(tgt_dir, 'r') as f2:
            f1 = f1.readlines()
            f2 = f2.readlines()
            if shuffle:
                random.seed(opt.seed)
                random.shuffle(f1)
                random.shuffle(f2)
            for i in range(len(f1)):
                s = self.tokenizer.encode(f1[i].strip()[:150])
                t = self.tokenizer.encode(f2[i].strip()[:150])
                src_seq.append(s)
                tgt_seq.append(t)

            return src_seq, tgt_seq


    def gen_loader(self, train_src, train_tgt, valid_src, valid_tgt):
        """Generate pytorch DataLoader."""

        train_loader = torch.utils.data.DataLoader(
            BartDataset(
                src_inst=train_src,
                tgt_inst=train_tgt),
            num_workers=2,
            batch_size=self.opt.batch_size,
            collate_fn=self.paired_collate_fn,
            shuffle=True)

        valid_loader = torch.utils.data.DataLoader(
            BartDataset(
                src_inst=valid_src,
                tgt_inst=valid_tgt),
            num_workers=2,
            batch_size=self.opt.batch_size,
            collate_fn=self.paired_collate_fn)

        return train_loader, valid_loader


    def collate_fn(self, insts):
        """Pad the instance to the max seq length in batch"""

        pad_id = self.tokenizer.pad_token_id
        max_len = max(len(inst) for inst in insts)

        batch_seq = [inst + [pad_id]*(max_len - len(inst))
                     for inst in insts]
        batch_seq = torch.LongTensor(batch_seq)

        return batch_seq


    def paired_collate_fn(self, insts):
        src_inst, tgt_inst = list(zip(*insts))
        src_inst = self.collate_fn(src_inst)
        tgt_inst = self.collate_fn(tgt_inst)

        return src_inst, tgt_inst


