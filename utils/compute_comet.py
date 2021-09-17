# -*- coding: utf-8 -*-

import os
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
from comet.models import download_model



def cal_comet(file_can, file_ref, num, model):
    """
    Calculate COMET score

    Args:
        file_can: the path of candidate file
        file_ref: the path of reference file
        num (int): the number of reference for each candidate
        model: COMET model

    Returns:
        the list of COMET scores
    """

    scores, srcs = [], []
    with open(file_can,'r') as fin:
        cand = []
        for line in fin.readlines():
            srcs.append('')
            cand.append(line.strip())

    for i in range(int(num)):
        refs = []
        with open(file1+str(i),'r') as fin:
            for line in fin.readlines():
                refs.append(line.strip())

        data = {"src": srcs, "mt": cand, "ref": refs}
        data = [dict(zip(data, t)) for t in zip(*data.values())]

        scores.extend(model.predict(data, cuda=True, 
                                    show_progress=False)[-1])

    return scores

scores = []
model = download_model("wmt-large-da-estimator-1719")
scores.extend(cal_comet(sys.argv[1], sys.argv[3], sys.argv[5], model))
scores.extend(cal_comet(sys.argv[2], sys.argv[4], sys.argv[5], model))
print('The average comet score is {}'.format(sum(scores)/len(scores)))
