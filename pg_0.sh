#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=50000

#python train_fpt.py -dataset sy -task ye -style 0
#python train_fpt.py -dataset sy -task ye -style 1
python train_ibt.py -dataset ye -order 0
python infer.py -dataset ye -order 0 -style 0
python infer.py -dataset ye -order 0 -style 1
python classifier/eval.py -dataset ye -order 0
python utils/tokenizer.py data/outputs/bart_ye_0.0 data/ye/outputs/bart_ye_0.0 False
python utils/tokenizer.py data/outputs/bart_ye_0.1 data/ye/outputs/bart_ye_0.1 False
perl utils/multi-bleu.perl data/ye/original_ref/pos.ref0 < data/ye/outputs/bart_ye_0.0
perl utils/multi-bleu.perl data/ye/original_ref/neg.ref0 < data/ye/outputs/bart_ye_0.1


#python train_fpt.py -dataset pp -task fr -style 0
#python train_fpt.py -dataset pp -task fr -style 1
python train_ibt.py -dataset fr -order 0
python infer.py -dataset fr -order 0 -style 0
python infer.py -dataset fr -order 0 -style 1
python classifier/eval.py -dataset fr -order 0
python utils/tokenizer.py data/outputs/bart_fr_0.0 data/ye/outputs/bart_fr_0.0 False
python utils/tokenizer.py data/outputs/bart_fr_0.1 data/ye/outputs/bart_fr_0.1 False
perl utils/multi-bleu.perl data/fr/original_ref/formal.ref < data/ye/outputs/bart_fr_0.0
perl utils/multi-bleu.perl data/fr/original_ref/informal.ref < data/ye/outputs/bart_fr_0.1
