import sys

"""
Select sentence-pairs based style classifier score and BLEURT score.

Args:
    sys.argv[1]: 1st input file with sentence-pairs and their BLEURT score (score sent1 sent2)
    sys.argv[2]: 2nd input file with source sentences and their style score (sent score)
    sys.argv[3]: 3rd input file with target sentences and their style score (sent score)
    sys.argv[4]: output file for source sentences
    sys.argv[5]: output fiel for target sentences
"""

data0 = open(sys.argv[1],'r').readlines()
data1 = open(sys.argv[2],'r').readlines()
data2 = open(sys.argv[3],'r').readlines()

with open(sys.argv[4],'w') as f0, open(sys.argv[5],'w') as f1:
    for l0, l1, l2 in zip(data0, data1, data2):
        l0 = l0.strip().split('\t')
        l1 = l1.split('\t')
        l2 = l2.split('\t')
        if ((float(l1[1])+float(l2[2]))/2>0.95 and float(l0[0])>0.15):
            f0.write(l0[1].strip()+'\n')
            f1.write(l0[2].strip()+'\n')
