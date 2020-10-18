"""
models.py

PURPOSE: This file defines Neural Network Architecture and other models
        which will be evaluated in this expirement
"""
import torch
import torch.nn as nn
from os import path


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def device(self):
        return next(self.parameters()).device

    def save_model(self):
        file_name = "{}.th".format(self.name)
        fn = path.join("ckpt", file_name)
        return torch.save(
            self.state_dict(),
            path.join(path.dirname(path.abspath(__file__)), fn),
        )

    def load_model(self):
        file_name = "{}.th".format(self.name)
        fn = path.join("ckpt", file_name)
        self.load_state_dict(
            torch.load(
                path.join(path.dirname(path.abspath(__file__)), fn),
                map_location=self.device(),
            )
        )


class GhostLSTM(BaseNetwork):
    """
    Defines the LSTM model based onGhostWriter Potash et al. 2015
    """

    def __init__(self, vocab_size, embedding_dim, hidden_size, name="ghost"):

        super().__init__()
        self.vocab_size = vocab_size
        self.name = name
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        # TODO: add args for
        self.encoder = nn.LSTM(embedding_dim, hidden_size)
        self.decoder = nn.LSTM(hidden_size, self.vocab_size)

    def forward(self, context):
        """
        Given a padded batch of input, with dimension [batch x seq_len],
        embeds & then passed through the network, producing [batch x out_len]
        where the output length is padded with <PAD>
        """
        pass


class CVAE(BaseNetwork):
    def __init__(self, vocab_size, embedding_dim, hidden_size, name="cvae"):

        super().__init__()
        self.vocab_size = vocab_size
        self.name = name

        self.embedding = nn.Embedding(self.vocab_size, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        # TODO: add args for
        self.encoder = nn.LSTM(embedding_dim, hidden_size)
        self.decoder = nn.LSTM(hidden_size, self.vocab_size)
