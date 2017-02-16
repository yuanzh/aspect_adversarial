## Running with Synthetic Data

### About
This directory contains the implementation of (1) generating random synthetic data for source and target domains and (2) running the aspect-augmented adversarial networks. 

The current implementation can generate four synthetic datasets that represent different challenges in domain adaptation. See [Synthetic Data](#synthetic-data) section for details of the datasets.

### Files
  - [data/generator.py](data/generator.py) generates different synthetic data.
  - [cnn_rel_adv_syn.py](cnn_rel_adv_syn.py) implements the aspect-augmented adversarial model for the synthetic data.

### Code Usage

#### Step 1: Generating the synthetic data

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
