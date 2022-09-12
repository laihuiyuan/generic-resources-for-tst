
# [Generic resources are what you need: Style transfer tasks without task-specific parallel training data (EMNLP 2021)](https://arxiv.org/pdf/2109.04543.pdf)


## Generic Resources
1. [PARABANK-2](http://cs.jhu.edu/~vandurme/data/parabank-2.0.zip)

2. [WordNet & SentiWordNet](https://www.nltk.org/) (via [NLTK](https://www/nltk/org/))

3. [BART Model](https://huggingface.co/models) (via [Transformers](https://huggingface.co/transformers/))

**Note:** we noticed that Hugging Face updated the configuration file, activation_dropout and attention_dropout in the previous version we used for the experiment were 0.0. To reproduce the experiment in the paper, you should change this setting.


## Dependencies
```
python==3.7
pytorch==1.4.0
transformers==2.5.1
nltk==3.4.5
```
In order to use BLEURT-based reward and evaluate the system, you may want to install [BLEURT](https://github.com/google-research/bleurt) and [COMET](https://github.com/Unbabel/COMET).

## Datasets
### [GYAFC](https://github.com/raosudha89/GYAFC-corpus): informal text (0) <-> formal text (1)
### [YELP](https://github.com/lijuncen/Sentiment-and-Style-Transfer/tree/master/data/yelp): negative text (0) <-> positive text (1)

## Quick Start

### Step 1: Pre-train Style Classifier
```
python classifier/textcnn.py -dataset fr 
```
**Note:** Our style classifiers are provided in checkpoints.


### Step 2: Generating Sentence-pairs from Generic Resources

Select sentence-pairs from PARABANK-2:
```
python utils/pair_parabank.py path_parabank2 path_source path_target
```

Generate sentence-pairs using WordNet
```
python utils/pair_wordnet.py path_input_file path_out_file_0 path_out_file_1 polarity (pos or neg)
```

Score sentence-paris using style classifier
```
python utils/score_textcnn.py -src path_source -tgt path_target -task task_name (fr or ye)
```

Select sentence-pairs based on style score
```
python utils/select_pair_sc.py source_input_file target_input_file source_out_file target_out_file
```


### Step 3 Further Pre-training: Learning to Rewrite
```
python train_fpt.py
```
**Note:** You may need to adjust the dataset path in utils/dataset.py. 


### Step 4: Iterative Back-translation and Rewards: Pairs on-the-fly
```
python train_ibt.py
```
**Note:** If you use BLEURT-based reward, please run this code with default parameters, otherwise BLERUT will report an error. In other words, you need to change the parameters in the script before running.


### Step 5: Final Training: High-quality Pairs
```
python infer.py
```
**Note:** You may need to adjust the path of non-parallel data to create a static resource of parallel data. 

Score and select sentence-pairs
```
python utils/score_bleurt.py path_source path_target path_out_file
python utils/score_textcnn.py -src path_source -tgt path_target -task task_name (fr or ye)
python utils/select_pair_sc_bleurt.py bleurt_file style_scource_file style_target_file out_source_file out_target_file
```

Finaly training with high-quality pairs and then generation
```
python train_fst.py
python infer.py
```

## System Outputs
The outputs of our best systems are provided in outputs.


## Citation
Please cite our EMNLP paper:
```
@inproceedings{lai-etal-2021-generic,
    title = "Generic resources are what you need: Style transfer tasks without task-specific parallel training data",
    author = "Lai, Huiyuan  and
      Toral, Antonio  and
      Nissim, Malvina",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.349",
    pages = "4241--4254",
}
```
