# -*- coding: utf-8 -*-

import os
import sys
import time
import math
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch import cuda
import torch.nn.functional as F
from transformers import BartTokenizer

sys.path.append("")
from utils.optim import ScheduledOptim

filter_sizes = [1, 2, 3, 4, 5]
num_filters = [128, 128, 128, 128, 128]

device = 'cuda' if cuda.is_available() else 'cpu'
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')


def load_embedding(tokenizer, embed_dim, embed_path=None):
    """
    Parse an embedding text file (e.g. Glove) into an torch.nn.Embedding layer.
    Args:
        tokenizer: BART tokenizer.
        embed_dim (int): word embedding dimension.
        embed_path: the path of embedding text file.
    Returns:
        embedding
    """

    embedding = np.random.normal(scale=embed_dim ** -0.5,
                                 size=(len(tokenizer), embed_dim))
    if embed_path == None:
        return embedding

    print('[Info] Loading embedding')
    embed_dict = {}
    with open(embed_path) as file:
        for i, line in enumerate(file):
            if i == 0:
                continue
            tokens = line.rstrip().split()
            try:
                embed_dict[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
            except:
                continue

    for i in range(len(tokenizer)):
        try:
            word = tokenizer.decode(i)
            if word in embed_dict:
                embedding[i] = embed_dict[word]
        except:
            print(i)

    return embedding


def evaluate(model, valid_loader, loss_fn, epoch):
    """
    Evaluation function for style classifier

    Args:
        model: TextCNN.
        valid_loader: Pytorch valid DataLoader.
        loss_fn: loss function
        epoch: the current training epoch.

    Returns:
        style accuracy, valid loss
    """

    model.eval()
    corre_num = 0.
    total_num = 0.
    loss_list = []
    with torch.no_grad():
        for batch in valid_loader:
            x_batch, y_batch = map(lambda x: x.to(device), batch)
            logits = model(x_batch)
            loss_list.append(loss_fn(logits, y_batch).item())
            _, y_hat = torch.max(logits,dim=-1)
            same = [float(p == q) for p, q in zip(y_batch, y_hat)]
            corre_num += sum(same)
            total_num += len(y_batch)
    model.train()
    print('[Info] Epoch {:02d}-valid | {}'.format(
         epoch, 'acc {:.2}% | loss {:.4f}').format(
         corre_num/total_num * 100, np.mean(loss_list)))

    return corre_num/total_num, np.mean(loss_list)

    
def collate_fn(insts):
    """
    Pad the instance to the max sequence length in batch.
    """

    max_len = max(len(inst) for inst in insts)
    max_len = max_len if max_len > 4 else 5

    batch_seq = np.array([
        inst + [tokenizer.pad_token_id] * (max_len - len(inst))
        for inst in insts])
    batch_seq = torch.LongTensor(batch_seq)

    return batch_seq


class TCNNDataset(torch.utils.data.Dataset):
    def __init__(self, insts, label):
        self.insts = insts
        self.label = label

    def __getitem__(self, index):
        return self.insts[index], self.label[index]

    def __len__(self):
        return len(self.insts)


def TCNNIterator(insts_neg, insts_pos, opt, shuffle=True):
    """
    Data iterator for style classifier
    Args:
        insts_neg (list): negative instances
        insts_pos (list): positive instances
    Returns:
        Pytorch DataLoader
    """

    def cls_fn(insts):
        insts, labels = list(zip(*insts))
        seq = collate_fn(insts)
        labels = torch.LongTensor(labels)

        return (seq, labels)

    num = len(insts_neg) + len(insts_pos)

    loader = torch.utils.data.DataLoader(
        TCNNDataset(
            insts=insts_neg + insts_pos,
            label=[0 if i < len(insts_neg)
                   else 1 for i in range(num)]),
        shuffle=shuffle,
        num_workers=2,
        collate_fn=cls_fn,
        batch_size=opt.batch_size)

    return loader


class EmbeddingLayer(nn.Module):
    """
    Args:
        vocab_size (int): vocabulary size
        embed_dim (int): word embedding dimension
        embedding (list): pre-trained word embedding
    """
    def __init__(self, vocab_size, embed_dim, embeding):
        super(EmbeddingLayer, self).__init__()
        self.embeding = nn.Embedding(vocab_size, embed_dim)
        if embeding is not None:
            self.embeding.weight.data = torch.FloatTensor(embeding)

    def forward(self, x):
        if len(x.size()) == 2:
            y = self.embeding(x)
        else:
            y = torch.matmul(x, self.embeding.weight)
        return y


class TextCNN(nn.Module):
    """
    A TextCNN Style Classifier
    Args:
        embed_dim (int): word embedding dimension
        vocab_size (int): vocabulary size
        filter_sizes (list): the size of filter
        num_filters (list): the number of filters
        embedding (list): pre-trained word embedding
        dropout (float): dropout
    """

    def __init__(self, embed_dim, vocab_size, filter_sizes, 
                 num_filters, embedding=None, dropout=0.0):
        super(TextCNN, self).__init__()

        self.feature_dim = sum(num_filters)
        self.embeder = EmbeddingLayer(vocab_size, embed_dim, embedding)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, n, (f, embed_dim))
            for (n, f) in zip(num_filters, filter_sizes)
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            self.dropout,
            nn.Linear(self.feature_dim, int(self.feature_dim / 2)), nn.ReLU(),
            nn.Linear(int(self.feature_dim / 2), 2)
        )

    def forward(self, inp):
        inp = self.embeder(inp).unsqueeze(1)
        convs = [F.relu(conv(inp)).squeeze(3) for conv in self.convs]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs]
        out = torch.cat(pools, 1)
        logit = self.fc(out)

        return logit

    def build_embeder(self, vocab_size, embed_dim, embedding=None):
        embeder = nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(embeder.weight, mean=0, std=embed_dim ** -0.5)
        if embedding is not None:
            embeder.weight.data = torch.FloatTensor(embedding)

        return embeder


def main():
    parser = argparse.ArgumentParser('TextCNN Style Classifier')
    parser.add_argument('-lr', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('-dataset', default='fr', type=str, help='the name of dataset')
    parser.add_argument('-embed_dim', default=300, type=int, help='the embedding size')
    parser.add_argument('-seed', default=42, type=int, help='pseudo random number seed')
    parser.add_argument("-dropout", default=0.5, type=float, help="Keep prob in dropout")
    parser.add_argument('-max_len', default=50, type=int, help='maximum tokens in a seq.')
    parser.add_argument('-log_step', default=100, type=int, help='print log every x steps')
    parser.add_argument('-eval_step', default=1000, type=int, help='early stopping training')
    parser.add_argument('-batch_size', default=32, type=int, help='maximum sents in a batch')
    parser.add_argument('-epoch', default=50, type=int, help='force stop at specified epoch')

    opt = parser.parse_args()
    torch.manual_seed(opt.seed)
    print('[Info]', opt)

    # load pre-trained word embedding
    if os.path.exists('checkpoints/embedding.pt'):
        embedding = torch.load('checkpoints/embedding.pt')
    else:
        embed_path = '../../data/glove.840B.300d.txt'
        embedding = load_embedding(tokenizer, 300, embed_path)
        torch.save(embedding, 'checkpoints/embedding.pt')

    # initialize the model
    model = TextCNN(opt.embed_dim, len(tokenizer), filter_sizes, 
                    num_filters, embedding, opt.dropout)
    model.to(device).train()
    print('[Info] Built a model with {} parameters'.format(
           sum(p.numel() for p in model.parameters())))

    # read instances from input file
    train_src, train_tgt, valid_src, valid_tgt = [], [], [], []
    with open('data/{}/train.0'.format(opt.dataset),'r') as f:
        for line in f.readlines():
            train_src.append(tokenizer.encode(line.strip())[:opt.max_len])
    with open('data/{}/train.1'.format(opt.dataset),'r') as f:
        for line in f.readlines():
            train_tgt.append(tokenizer.encode(line.strip())[:opt.max_len])
    with open('data/{}/valid.0'.format(opt.dataset),'r') as f:
        for line in f.readlines():
            valid_src.append(tokenizer.encode(line.strip())[:opt.max_len])
    with open('data/{}/valid.1'.format(opt.dataset),'r') as f:
        for line in f.readlines():
            valid_tgt.append(tokenizer.encode(line.strip())[:opt.max_len])
    print('[Info] {} instances from train set'.format(len(train_src)))
    print('[Info] {} instances from valid set'.format(len(valid_tgt)))

    train_loader = TCNNIterator(train_src, train_tgt, opt)
    valid_loader = TCNNIterator(valid_src, valid_tgt, opt)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = ScheduledOptim(
        torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                         betas=(0.9, 0.98), eps=1e-09), opt.lr)

    tab = 0
    avg_acc = 0
    corre_num = 0.
    total_num = 0.
    loss_list = []
    start = time.time()
    for e in range(opt.epoch):

        for idx, batch in enumerate(train_loader):
            x_batch, y_batch = map(lambda x: x.to(device), batch)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = loss_fn(logits, y_batch)
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()

            y_hat = logits.argmax(dim=-1)
            same = [float(p == q) for p, q in zip(y_batch, y_hat)]
            corre_num += sum(same)
            total_num += len(y_batch)

            if optimizer.steps % opt.log_step == 0:
                lr = optimizer._optimizer.param_groups[0]['lr']
                print('[Info] Epoch {:02d}-{:05d}: | acc {:.2f}% | '
                    'loss {:.4f} | lr {:.6f} | second {:.2f}'.format(
                    e, optimizer.steps, corre_num / total_num * 100,
                    np.mean(loss_list), lr, time.time() - start))
                corre_num = 0.
                total_num = 0.
                loss_list = []
                start = time.time()

            if optimizer.steps % opt.eval_step == 0:
                valid_acc, valid_loss = evaluate(model, valid_loader, loss_fn, e)
                if avg_acc < valid_acc:
                    avg_acc = valid_acc
                    save_path = 'checkpoints/textcnn_{}.chkpt'.format(opt.dataset)
                    torch.save(model.state_dict(), save_path)
                    print('[Info] The checkpoint file has been updated.')
                    tab = 0
                else:
                    tab += 1
                    if tab == 5:
                        exit()


if __name__ == '__main__':
    main()

