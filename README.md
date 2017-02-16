## Aspect-augmented Adversarial Networks for Domain Adaptation

### About this repo
This repo contains the code and resources of the following paper:

  - [Aspect-augmented Adversarial Networks for Domain Adaptation](https://arxiv.org/pdf/1701.00188.pdf). Yuan Zhang, Regina Barzilay, and Tommi Jaakkola.

This paper introduces an adversarial network method for transfer learning between two (source and target) classification tasks or aspects over the same domain.

### Structures
  - [/nn](nn): source code of NN library based on this [repo](https://github.com/taolei87/rcnn/tree/master/code/nn)
  - [/word2vec](word2vec): word2vec (from [here](https://code.google.com/archive/p/word2vec/)) for training word embeddings
  - [/synthectic](synthetic): source code for generating and training models on synthetic data

### Dependencies
  [Theano](http://deeplearning.net/software/theano/) >= 0.8, Python >= 2.7, Numpy

### To-do
  - [ ] source code for the review dataset
