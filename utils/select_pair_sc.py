import sys

"""
Choose sentence-pairs based style classifier score

Args:
    sys.argv[1]: 1st input file with source sentences and their style score
    sys.argv[2]: 2nd input file with target sentences and their style score
    sys.argv[3]: output file for source sentences
    sys.argv[4]: output fiel for target sentences
"""

f3 = open(sys.argv[3],'w')
f4 = open(sys.argv[4],'w')

with open(sys.argv[1],'r') as f1, open(sys.argv[2], 'r') as f2:
    for l1,l2 in zip(f1.readlines(), f2.readlines()):
        l1 = l1.split('\t')
        l2 = l2.split('\t')
        if (float(l1[1])+float(l2[2]))/2>0.85:
            f3.write(l1[0].strip()+'\n')
            f4.write(l2[0].strip()+'\n')
        elif (float(l1[2])+float(l2[1]))/2>0.85:
            f3.write(l2[0].strip()+'\n')
            f4.write(l1[0].strip()+'\n')

