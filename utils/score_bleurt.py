# -*- coding: utf-8 -*-


import os
import sys
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from bleurt import score

"""
Score sentence-pairs by using BLEURT

Args:
    sys.argv[1]: input file with source sentences
    sys.argv[2]: input file with target sentences
    sys.argv[3]: output file
"""

checkpoint = 'checkpoints/bleurt-base-128'
scorer = score.BleurtScorer(checkpoint)

sents = []
with open(sys.argv[1],'r') as f0, \
     open(sys.argv[2],'r') as f1, \
      open(sys.argv[3],'w') as f2:
    for l0, l1 in zip(f0.readlines(),f1.readlines()):
        hyps, refs = [],[]
        hyps.append(l0.strip())
        refs.append(l1.strip())
        scores = scorer.score(refs, hyps)
        line = str(round(scores[0],5))+'\t'+l0.strip()+'\t'+l1.strip()+'\n'
        f2.write(line)
