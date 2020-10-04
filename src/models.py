"""
models.py

PURPOSE: This file defines Neural Network Architecture and other models
        which will be evaluated in this expirement
"""
import torch
import torch.nn as nn
from os import path


class GhostLSTM(nn.Module):
    """
    Defines the LSTM model based onGhostWriter Potash et al. 2015
    """

    def __init__(self, vocab_size, embedding_dim, hidden_size):

        super().__init__()
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(self.vocab_size, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        # TODO: add args for
        self.encoder = nn.LSTM(embedding_dim, hidden_size)
        self.decoder = nn.LSTM(hidden_size, self.vocab_size)

    def device(self):
        if next(self.parameters()).is_cuda:
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def forward(self, context):
        """
        Given a padded batch of input, with dimension [batch x seq_len],
        embeds & then passed through the network, producing [batch x out_len]
        where the output length is padded with <PAD>
        """
        pass


def save_model(model):
    """
    Saves the model to a specified location
    """
    if isinstance(model, GhostLSTM):
        return torch.save(
            model.state_dict(),
            path.join(path.dirname(path.abspath(__file__)), "ghost.th"),
        )
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    """
    Loads a saved model, depending on the type of model
    """
    if isinstance(model, GhostLSTM):
        model.load_state_dict(
            torch.load(
                path.join(path.dirname(path.abspath(__file__)), "ghost.th"),
                map_location=model.device(),
            )
        )
        return model
    else:
        raise ValueError("model type '%s' not supported!" % str(type(model)))
