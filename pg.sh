#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=50000

python train_fpt.py -dataset sy -task ye -style 0
python train_fpt.py -dataset sy -task ye -style 1
python train_ibt.py -dataset $2 -order $1
python infer.py -dataset $2 -order $1 -style 0
python infer.py -dataset $2 -order $1 -style 1
python classifier/eval.py -dataset $2 -order $1
python utils/tokenizer.py data/outputs/bart_$2_$1.0 data/$2/outputs/bart_$2_$1.0 False
python utils/tokenizer.py data/outputs/bart_$2_$1.1 data/$2/outputs/bart_$2_$1.1 False
perl utils/multi-bleu.perl data/$2/original_ref/pos.ref0 < data/$2/outputs/bart_$2_$1.0
perl utils/multi-bleu.perl data/$2/original_ref/neg.ref0 < data/$2/outputs/bart_$2_$1.1
