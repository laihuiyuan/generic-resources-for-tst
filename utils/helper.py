# -*- coding: utf-8 -*-

import math
import nltk
import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

from torch import cuda
import torch.nn as nn
import torch.nn.functional as F

from classifier.textcnn import collate_fn

device = 'cuda' if cuda.is_available() else 'cpu'


def optimize(opt, loss, retain_graph=False):
    """optimize function"""

    opt.zero_grad()
    loss.backward(retain_graph=retain_graph)
    opt.step()


def cal_reward_loss(sample_probs, reward, lens=None):
    """
    Reward based loss

    Args:
        sample_probs: the sample probability of each id in the sequence
        reward: the reward for each sequence
        lens: the 'true' lenght of each sequence (except the pad_id)
    Returns:
        loss
    """

    sample_probs = sample_probs.contiguous()
    sample_logprobs = torch.log(sample_probs)
    reward = reward.unsqueeze(1).contiguous()
    if lens is not None:
        batch_size, max_len = sample_probs.size()
        mask = torch.zeros(batch_size, max_len).to(device)
        for i, l in enumerate(lens):
            mask[i, :l] = 1
        mask = mask.float().contiguous()
        output = -sample_logprobs * reward * mask
        output = (output.sum(-1) / mask.sum(-1)).mean()
    else:
        output = -sample_logprobs * reward
        output = output.mean()

    return output


def cal_sc_loss(outs, lens, clssifier, eos_id, style, softmax=True):
    """
    Style classifier based reward

    Args:
        outs: the output of transferring model
        lens: the 'true' lenght of each sequence (except the pad_id)
        classifier: style classifier
        eos_od: eos token id
        style: transfer direction
        softmax: whether do softmax
    """

    if softmax:
        outs = F.softmax(outs, dim=-1)
    
    sample_probs, sample_idx = sample_3d(outs)

    if lens == None:
        lens = outs.new(outs.size(0), 1).fill_(outs.size(1)).long()

    seqs, idxs = [], []
    for s, l in zip(sample_idx, lens.cpu()):
        e = torch.arange(len(s))[s.eq(eos_id)]
        e = e[0] if 0 < len(e) and 4 < e[0] < l else l
        seqs.append(s[:e].cpu().tolist())
        idxs.append(e)
    seq = collate_fn(seqs).to(device)
    with torch.no_grad():
        cls = F.softmax(clssifier(seq), -1)

    if style == 0:
        reward = cls[:, 1] - cls[:, 0]
    else:
        reward = cls[:, 0] - cls[:, 1]

    loss_sc = cal_reward_loss(sample_probs, reward, lens.cpu())

    return loss_sc


def cal_bleu_loss(outs, refs, lens, eos_id, softmax=True):
    """
    BLEU based reward

    Args:
        outs: the output of transferring model
        refs: the human reference
        eos_od: eos token id
        softmax: whether do softmax
    """

    if softmax:
        outs = F.softmax(outs, dim=-1)
    smooth = SmoothingFunction()
    sample_prob, sample_idx = sample_3d(outs)

    reward, idxs = [], []
    for s, r, l in zip(sample_idx, refs, lens):
        e = torch.arange(len(s))[s.eq(eos_id)]
        e = e[0] if 0 < len(e) and 4 < e[0] < l else l
        hpy = s[1:e].cpu().tolist()
        ref = [r[1:l].cpu().tolist()]
        idxs.append(e)
        reward.append(sentence_bleu(ref, hpy,
                      smoothing_function=smooth.method1))
    reward = torch.FloatTensor(reward).to(device)
    loss_co = cal_reward_loss(sample_prob, reward, idxs)

    return loss_co


def cal_bleurt_loss(outs, refs, lens, tokenizer, model, softmax=True):
    """
    BLEURT based reward

    Args:
        outs: the output of transferring model
        refs: the human reference
        tokenizer: BART tokenizer
        model: BLEURT 
        softmax: whether do softmax
    """

    if softmax:
        outs = F.softmax(outs, dim=-1)
    sample_probs, sample_idx = sample_3d(outs)

    hyp, ref, idxs = [], [], []
    for l, s, t in zip(lens.cpu(), sample_idx, refs):
        e = torch.arange(len(s))[s.eq(tokenizer.eos_token_id)]
        e = e[0] if 0 < len(e) and 4 < e[0] < l else l
        hyp.append(tokenizer.decode(s[1:e].cpu().tolist()))
        ref.append(tokenizer.decode(t[1:l].cpu().tolist()))
        idxs.append(e)
    reward = model.score(ref, hyp)
    reward = torch.FloatTensor(reward).to(device)
    loss_co = cal_reward_loss(sample_probs, reward, idxs)
    
    return loss_co


def sample_3d(probs, temperature=1):
    """probs.shape = (batch, seq_len, dim)"""

    sample_idx = probs.new(probs.size()[:2]).fill_(0).long()
    sample_pro = probs.new(probs.size()[:2]).fill_(0).float()
    if temperature != 1:
        temp = torch.exp(torch.div(torch.log(probs + 1e-20), temperature))
    else:
        temp = probs
    for i, s in enumerate(temp):
        temp_idx = torch.multinomial(s, 1)  # shape = (seq_len, 1)
        temp_probs = s.gather(1, temp_idx)  # shape = (seq_len, 1)
        sample_idx[i] = temp_idx.squeeze(1)
        sample_pro[i] = temp_probs.squeeze(1)

    return sample_pro, sample_idx

