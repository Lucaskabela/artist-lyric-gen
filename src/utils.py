# Taken from
# https://github.com/pytorch/examples/blob/master/word_language_model/data.py
import os
import torch
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


class Dictionary(object):
    def __init__(self):
        self.word2idx = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.idx2word = ["<pad>", "<sos>", "<eos>", "<unk>"]
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
        # Get and add personas, as well as tokenize the personas
        self.train = self.tokenize(os.path.join(path, "train.txt"))
        self.valid = self.tokenize(os.path.join(path, "valid.txt"))
        self.test = self.tokenize(os.path.join(path, "test.txt"))

    def tokenize(self, path, max_context=100):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, "r", encoding="utf8") as f:
            lyrics = json.load(f)
            prev_line = ""
            for verse in lyrics:
                words = line.split() + ["<eos>"]
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        # Should be tuple (previous lines, persona, current line)
        # Might consider limiting pervious lines/size too!
        with open(path, "r", encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ["<eos>"]
                if len(words) > 1:
                    ids = []
                    for word in words:
                        ids.append(self.dictionary.word2idx[word])
                    idss.append(torch.tensor(ids).type(torch.long))
        # Do not return idss, return the datasets instead!
        return idss


# Should return something we can get batches from - dataloader?
def load_data(dataset, batch_size=256, num_workers=1):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=collate_pad_sentences,
    )


class RapPersonaDataset(Dataset):
    """
    A dataset containing target sentences, previous lines, and personas
    There is no target (for training reconstruction loss)
    """

    def __init__(self, persona_idxs, personas, prev, sents):
        self.persona_idxs = persona_idxs
        self.personas = personas
        self.prev = prev
        self.sents = sents

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, index):
        persona =  self.personas[self.persona_idxs[index]]
        return self.prev[index], persona, self.sents[index]


# a simple custom collate function, just put them into a list!
def collate_pad_sentences(batch):
    prev, persona, line = batch
    prev_lens = torch.LongTensor([len(x) for x in prev])
    prev_pad = pad_sequence(prev, batch_first=True, padding_value=0)
    persona_lens = torch.LongTensor([len(x) for x in persona])
    persona_pad = pad_sequence(persona, batch_first=True, padding_value=0)
    batch_lens = torch.LongTensor([len(x) for x in line])
    batch_pad = pad_sequence(line, batch_first=True, padding_value=0)
    return prev_pad, prev_lens, persona_pad, persona_lens, batch_pad, batch_lens
