# -*- coding: utf-8 -*-

import os
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
from bleurt import score


def cal_bleurt(file_can, file_ref, num, model):
    """
    Calculate BLEURT score between candidate and reference

    Args:
        file_can: the path of candidate file
        file_ref: the path of reference file
        num (int): the number of reference for each candidate
        model: BLEURT model

    Returns:
        the list of BLEURT scores
    """

    cand, scores = [], []
    with open(file_can,'r') as fin:
        cand = []
        for line in fin.readlines():
            cand.append(line.strip())

    for i in range(int(num)):
        refs = []
        with open(file_ref+str(i),'r') as fin:
            for line in fin.readlines():
                refs.append(line.strip())
            scores.extend(model.score(refs, cand))

    return scores


scores = []
checkpoint = 'checkpoints/bleurt-base-128'
scorer = score.BleurtScorer(checkpoint)
scores.extend(cal_bleurt(sys.argv[1], sys.argv[3], sys.argv[5], scorer))
scores.extend(cal_bleurt(sys.argv[2], sys.argv[4], sys.argv[5], scorer))
print('The average bleurt score is {}'.format(sum(scores)/len(scores)))
