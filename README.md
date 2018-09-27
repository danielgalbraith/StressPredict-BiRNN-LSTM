# StressPredict-BiRNN-LSTM

Multi-class classification via bi-directional recurrent neural network with LSTM layers and a 1D convolution. Predicts sentence stress annotations, trained on data from US Presidential Inaugural Addresses with gold-standard corpus. Uses Keras Sequential model in Python 3.

* **Input features**: Word embedding matrices from training texts and pre-trained GloVe 100-dimensional word vectors (GloVe vectors not uploaded, see https://nlp.stanford.edu/projects/glove/ for download).
* **Output**: 7 possible annotations {0,1,...,6}, representing relative stress prominence.
* **Unseen data**: Additional unseen text of Franklin D. Roosevelt's State of the Union address, delivered Jan 11, 1944 (public domain).

Training data not uploaded; consult Anttila et al. (2017) for further information on the annotation procedure.

Based on the following research:

Anttila, Arto, Timothy Dozat, Daniel Galbraith and Naomi Shapiro. 2017. Sentence stress in presidential speeches. *The 39th Annual Meeting of the DGfS Workshop on Prosody in Syntactic Encoding*, Saarbrücken, March 9, 2017.
