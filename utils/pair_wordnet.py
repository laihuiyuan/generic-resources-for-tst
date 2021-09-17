# -*- coding: utf-8 -*-

"""Generate sentence-pairs by using WordNet"""

import sys
import nltk
import string
import copy
#nltk.download('sentiwordnet')
#nltk.download('averaged_perceptron_tagger')
import random
from nltk.corpus import wordnet
from nltk.corpus import sentiwordnet as swn
from nltk.tokenize.treebank import TreebankWordDetokenizer


punc = string.punctuation
punc += "n't，《。》、？；：‘”【「】」、|·~！@#￥%……&*（）-——+='s'm'll''``"

def detokenize(tokens):
    """
    The detokenizer that is used to generate the sentence from tokens
    Args:
        tokens (list): the list of tokens
    Returns:
        the string of a sequence
    """

    line = TreebankWordDetokenizer().detokenize(tokens)
    tokens = line.split()
    line = ''
    for token in tokens:
        if token not in punc:
            line += (' '+token)
        else:
            line += token
    return line.strip()


def check(word, pola='pos'):
    """
    Check it to see if it is a polar word
    Args:
       word (str): a word
       pola: the polarity
    Returns:
       bool
    """

    synset_forms = list(swn.senti_synsets(word))
    if not synset_forms:
        return False

    synset = synset_forms[0] 
    pos_score = synset.pos_score()
    neg_score = synset.neg_score()
    if (pola=='pos' and pos_score>0 and neg_score==0)\
        or (pola=='neg' and neg_score>0 and pos_score==0):
        return True
    else:
        return False


def replace(word):
    """
    Replace the polar word with their antonym
    Args:
        word (str): a word
    Returns:
        the antonym of the input word
    """
    syn = list()
    ant = list()
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            syn.append(lemma.name())
            if lemma.antonyms():
                w_pos = lemma.antonyms()[0].synset().pos()
                w = lemma.antonyms()[0].name()
                ant.append((w,w_pos))
    if len(ant)!=0:
#        return ant[0][0]
        return random.sample(ant,1)[0][0]

    return word

"""
Args:
    sys.argv[1]: input file
    sys.argv[2]: output file with original sentences
    sys.argv[3]: output file with changed sentences
    sys.argv[4]: polarity (pos for positive, neg for negative)
"""
f0 = open(sys.argv[1],'r').readlines()

with open(sys.argv[2],'w') as f3, open(sys.argv[3],'w') as f4:
    for j, l0 in enumerate(f0):
        sents = []
        sents.append(l0.strip())
        token_0 = nltk.word_tokenize(l0.strip())
        token_1 = copy.deepcopy(token_0)
        pos_tags =nltk.pos_tag(token_0)
        for i in range(len(token_0)):
            pola = check(token_0[i], sys.argv[4])
            if pola:
                word = replace(token_0[i])
                if word!=token_0[i]:
                    temp = copy.deepcopy(token_0)
                    temp[i] = word
                    sents.append(detokenize(temp))
                    token_1[i] = word

        if len(sents)>2:
            sents.append(detokenize(token_1))
        if len(sents)==2:
           f3.write(l0.strip()+'\n')
           f4.write(sents[-1].strip()+'\n')
