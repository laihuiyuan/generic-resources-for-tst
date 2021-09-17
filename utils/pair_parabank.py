import sys

"""
Choose sentence-pairs based on the bidirectional model scores and sentence length.

Args:
    sys.argv[1]: input file of parabank2
    sys.argv[2]: output file for source sentences
    sys.argv[3]: output fiel for target sentences
"""

with open(sys.argv[1],'r') as f0, \
     open(sys.argv[2],'w') as f1, \
     open(sys.argv[3],'w') as f2:
    for line in f0.readlines():
        line = line.strip().split('\t')
        if float(line[0])>0.031 or len(line)<3:
            continue
        len0 = len(line[1].split())
        len1 = len(line[2].split())
        if 5<len0 and len0<40 and 5<len1 and len1<40:
            f1.write(line[1].strip()+'\n')
            f2.write(line[2].strip()+'\n')

