# artist-lyric-gen
A research project in conditional lyric generation conditioned on artists

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


## Prerequisites

This project uses conda to manage packages and requirements - however pip can also be used.

+ [conda](https://docs.anaconda.com/anaconda/install/)

+ [pip](https://pip.pypa.io/en/stable/installing/)

See setup.sh to get started and install requirements for the project

## Dataset

Important files relating to the dataset (train splits, artist list, token lists,
etc) can be found in
[src/dataset](https://github.com/Lucaskabela/artist-lyric-gen/tree/master/src/dataset)

The dataset can be recreated from following the notebook
[src/dataset/dataset.ipynb](https://github.com/Lucaskabela/artist-lyric-gen/blob/master/src/dataset/dataset.ipynb)

## BPE

We forked Sennrich's BPE implementation
[here](https://github.com/billyang98/subword-nmt) and BPE can be run using the
[notebook](https://github.com/billyang98/subword-nmt/blob/94f31078df120b260b242124ee35accb559c8491/running_bpe.ipynb)

## Training

For CVAE, we have a [training notebook](https://github.com/Lucaskabela/artist-lyric-gen/blob/master/src/notebooks/cvae_train.ipynb) which demonstrates the commands needed to train the model and generate verses

For BART, we have a training repo [artist-lyric-gen-bart](https://github.com/billyang98/artist-lyric-gen-bart)


## Evaluation

For BART, generation and perplexity (things needing the BART model) can be done
following our notebook [src/Generation + Perplexity ... .ipynb](https://github.com/Lucaskabela/artist-lyric-gen/blob/master/src/Generation_%2B_Perplexity_(BART_Learns_to_Rap_Medium).ipynb)

All other metrics using the generated verses can be done following
[src/artist_gen_eval.ipynb](https://github.com/Lucaskabela/artist-lyric-gen/blob/master/src/artist_gen_eval.ipynb)

