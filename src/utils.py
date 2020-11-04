# Taken from
# https://github.com/pytorch/examples/blob/master/word_language_model/data.py
import json
import os
import torch
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
# from data.persona_parser import create_personas
# from dataset.dataset_utils import apply_bpe_to_string, get_bpe_object, clean_artist_names

class Dictionary(object):
    def __init__(self):
        self.word2idx = {"<pad>": 0, "S": 1, "L": 2, "<unk>": 3}
        self.idx2word = ["<pad>", "S", "L", "<unk>"]
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

    def __init__(self, path, persona_path):
        self.dictionary = Dictionary()
        # Get and add personas, as well as tokenize the personas
        # self.bpe = get_bpe_object()
        # self.personas = self.tokenize_p(create_personas(persona_path))
        self.personas = self.tokenize_p_2(persona_path)
        self.train = self.tokenize(os.path.join(path, "train.json"))
        self.valid = self.tokenize(os.path.join(path, "val.json"))
        self.test = self.tokenize(os.path.join(path, "test.json"))

    def tokenize_p_2(self, personas_path):
        res = {}
        idx = 1
        with open(personas_path, 'r') as personas:
            for line in personas:
                words = line.split()
                for word in words:
                    self.dictionary.add_word(word)
                idxs = []
                for word in words:
                    idxs.append(self.dictionary.word2idx[word])
                res[idx] = idxs
                idx += 1
        return res

    def tokenize_p(self, personas):
        natural = False
        res = {}
        for name in personas:
            person = personas[name]
            
            if natural:
                fn = person.to_natural_input
            else:
                fn = person.to_nn_input
            persona_text = fn(use_rn=True, use_city=True, use_nn=True, use_group=True, use_discog=True, use_year=True)
            persona_text = clean_artist_names(persona_text)
            persona_text = apply_bpe_to_string(persona_text, self.bpe)
            words = persona_text.split()
            for word in words:
                self.dictionary.add_word(word)
            idxs = []
            for word in words:
                idxs.append(self.dictionary.word2idx[word])
            res[person.id] = idxs
        return res

    def tokenize(self, path, max_context=100):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, "r", encoding="utf8") as f:
            songs = json.load(f)
            for verse in songs:
                lyrics = verse["lyrics"].split("L")
                for line in lyrics:
                    # Get each line in the song
                    words = line.split() + ["L"]
                    for word in words:
                        self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, "r", encoding="utf8") as f:
            x_s = []
            personas = []
            y_s = []
            songs = json.load(f)
            for song in songs:
                artist = song["artist_id"]
                artist_persona = self.personas[artist]
                verse = song["lyrics"].split("L")
                context = [1] # Place holder, make song title
                for line in verse:
                    words = line.split()[1:] # Disregard start of line
                    ids = []
                    for word in words:
                        ids.append(self.dictionary.word2idx[word])
                    x_s.append(torch.tensor(context[-max_context:]).long())
                    personas.append(torch.tensor(artist_persona).long())
                    y_s.append(torch.tensor(ids).type(torch.long))
                    # Do not append start of sequence to context
                    context.extend(ids[1:])
        # Do not return idss, return the datasets instead!
        return RapPersonaDataset(x_s, personas, y_s)


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

    def __init__(self, prev, personas, sents):
        self.prev = prev
        self.personas = personas
        self.sents = sents
        assert len(self.prev) == len(self.sents)
        assert len(self.personas) == len(self.sents)

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, index):
        return self.prev[index], self.personas[index], self.sents[index]


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
