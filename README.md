
# [Generic resources are what you need: Style transfer tasks without task-specific parallel training data (EMNLP 2021)]()

Code coming soon.

## Generic Resources
1. [BART Model](https://huggingface.co/models)

2. [PARABANK-2](http://decomp.io/projects/parabank2/)

3. [WordNet](https://www.nltk.org/) (using NLTK to download)

## Dependencies
```
python==3.7
pytorch==1.4.0
transformers==2.5.1
nltk==3.4.5
```
For evaluation, you may want to install [BLEURT](https://github.com/google-research/bleurt) and [COMET](https://github.com/Unbabel/COMET).

## Dataset
### [GYAFC](https://github.com/raosudha89/GYAFC-corpus): informal text (0) <-> formal text (1)
### [YELP](https://github.com/lijuncen/Sentiment-and-Style-Transfer/tree/master/data/yelp): negative text (0) <-> positive text (1)

## Quick Start

### Step 1: Pre-train style classifier
```
python classifier/textcnn.py -dataset fr
```
**Note:** Our style classifiers are provided in checkpoints.

### Step 2: Generating sentence-pairs from generic resources

Select sentence-pairs from PARABANK-2:
```
python utils/pair_parabank.py path_parabank2 path_source path_target
```

Score sentence-paris using style classifier
```
python utils/score_textcnn.py -src path_source -tgt path_target -task fr (ye)
```

Select sentence-pairs based on style score
```
python utils/select_pair_sc.py source_input_file target_input_file source_out_file target_out_file
```

Generate sentence-pairs using WordNet
```
python utils/pair_wordnet.py path_input_file path_out_file_0 path_out_file_1
```

### Step 3 Further Pre-training: Learning to Rewrite
```
python train_fur.py
```
**Note:** You may need to adjust the dataset path in utils/dataset.py. 


### Step 4: Iterative Back-translation and Rewards: Pairs on-the-fly
```
python train_ibt.py
```

### Step 5: Final Training: High-quality Pairs
```
python infer.py
```
**Note:** You may need to adjust the path of non-parallel data to create a static resource of parallel data. 

Score and select sentence-pairs
```
python utils/score_bleurt.py path_source path_target path_out_file
python utils/score_textcnn.py -src path_source -tgt path_target -task fr (ye)
python utils/select_pair_sc_bleurt.py bleurt_file style_scource_file style_target_file out_source out_target
```

Finaly traing with high-quality pairs
```
python train_sup.py
```

Generation
```
python infer.py
```

## System Output
The outputs of our best systems are provided in outputs.

