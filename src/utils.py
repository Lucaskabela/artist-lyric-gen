# Taken from
# https://github.com/pytorch/examples/blob/master/word_language_model/data.py
import os
import torch
from collections import Counter, OrderedDict
from torch.utils.data import Sampler, Dataset, DataLoader
from random import shuffle


class Dictionary(object):
    def __init__(self):
        self.word2idx = {"<PAD": 0, "<SOS>": 1, "<EOS>": 2}
        self.idx2word = ["<PAD", "<SOS>", "<EOS>"]
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    """
    self.train, valid, and test are List[Tensor] where tensors are
        (sent_len)
    dimensions.  Use these later to batch
    """

    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, "train.txt"))
        self.valid = self.tokenize(os.path.join(path, "valid.txt"))
        self.test = self.tokenize(os.path.join(path, "test.txt"))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                words = line.split() + ["<eos>"]
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, "r", encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ["<eos>"]
                if len(words) > 1:
                    ids = []
                    for word in words:
                        ids.append(self.dictionary.word2idx[word])
                    idss.append(torch.tensor(ids).type(torch.long))
        return idss


# Should return something we can get batches from - dataloader?
def load_data(dataset, batch_size=256, num_workers=4):
    dat = SentDataset(dataset)
    return DataLoader(
        dat,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_sentences,
    )


class SentDataset(Dataset):
    """
    A dataset containing target sentences - just used for batching
    and sampling.  There is no target (for training reconstruction)
    """

    def __init__(self, sents):
        self.sents = sents

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, index):
        return self.sents[index]


# a simple custom collate function, just put them into a list!
def collate_sentences(batch):
    data = [item for item in batch]
    return data
