## Running with Synthetic Data

### About
This directory contains the implementation of (1) generating random synthetic data for source and target domains and (2) running the aspect-augmented adversarial networks. 

The current implementation can generate four synthetic datasets that represent different challenges in domain adaptation. See [Synthetic Data](#synthetic-data) section for details of the datasets.

### Files
  - [data/generator.py](data/generator.py) generates different synthetic data.
  - [cnn_rel_adv_syn.py](cnn_rel_adv_syn.py) implements the aspect-augmented adversarial model for the synthetic data.

### Code Usage
First, clone the repo and make sure Numpy and Theano are installed. Next, follow the instructions below.

#### Step 1: Generating the synthetic data
Example run.
```
export THEANO_FLAGS='device=gpu,floatX=float32'     # use GPU and 32-bit float
python data/generator.py                            # synthetic data generator
      --mode=0                                      # which dataset to generate (required, 0 to 3)
```
It will automatically generate the data in the [data](data) directory and then run word2vec to generate word embeddings.

#### Step 2: Run the adversarial model on the synthtic data
Assume that we generate the data with ``mode=0``, we can run the model as follows:
```
python cnn_rel_adv_syn.py                                # model implementation 
      --source_train=data/syn0.source.train              # path to source training set (required)
      --source_unlabel=data/syn0.source.ul               # path to source unlabeled set (required)
      --target_unlabel=data/syn0.target.ul               # path to target unlabeled set (required)
      --embeddings=data/syn0.emb.100                     # path to word embeddings (required)
      --dev=data/syn0.dev                                # path to development set (required)
      --test=data/syn0.test                              # path to testing set (required)
      --batch=64                                         # minibatch size
      --rho=0.5                                          # strength of adversarial training
```
Use ``python cnn_rel_adv_syn.py --help`` to see more options.

#### (Optional) Step 3: Run the model without adversarial training
Simply set ``rho=0.0`` and the model will train without the adversarial component.

### Synthetic Data
Synthetic datasets as follows are generated as follows.

  1. Each synthetic document consists of around ten randomly generated sentences. 
  2. Each sentence is always associated with a random aspect and contains a special token as the aspect name (e.g. ASP0_NAME0). 
  3. Except for the first dataset, each sentence also contains another special token as the aspect polarity (e.g. ASP0_POS_NAME0 or POS_NAME0). 
  4. Aspect names and polarity tokens each have about ten different options (i.e. NAME0 to NAME9). 
  5. Document labels are either positive or negative, indicated by the polarity tokens of the focal aspect, except for the first dataset (see below). 
  
The adaptation task is to transfer the model from one aspect to another. The characteristics of each dataset are as follows.

  - **SYN1**: Sentences do not contain polarity tokens. Instead, class labels are indicated by the occurrence of aspect names. The label is positive if a name of the particular aspect (e.g. ASP0_NAME0) occurs, otherwise negative.\  
  - **SYN2**: Class labels are indicated by polarity tokens. To make the transfer a possible task, positive polarity tokens have 20% overlap across aspects. In other words, for 20% of the sentences with positive polarity tokens, the tokens have the format POS_NAME0 while the rest have the format ASP0_POS_NAME0. In contrast, negative polarity tokens have no overlap.
  - **SYN3**: Both positive and negative polarity tokens have 20% overlap across aspects.
  - **SYN4**: The last dataset is similar to the third one. However, both positive and negative polarity tokens have only 5% overlap across aspects.

To distinguish aspect names and polarity words from others, we surround each of them with a different distribution of context words. We fill in the rest place of sentences with other random words. See generated files for examples.

### Result
