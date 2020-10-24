"""
models.py

PURPOSE: This file defines Neural Network Architecture and other models
        which will be evaluated in this expirement
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from os import path
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pathlib


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def device(self):
        return next(self.parameters()).device

    def save_model(self):
        dir = "ckpt"
        pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
        file_name = "{}.th".format(self.name)
        fn = path.join(dir, file_name)
        return torch.save(
            self.state_dict(),
            path.join(path.dirname(path.abspath(__file__)), fn),
        )

    def load_model(self):
        file_name = "{}.th".format(self.name)
        fn = path.join("ckpt", file_name)
        if not path.exists(fn):
            raise Exception("Missing saved model")
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
    """
    Defines the CVAE approach, using LSTMs as Encoder/Decoder
    """

    def __init__(
        self,
        vocab,
        emb_dim,
        hidden_size,
        latent_dim,
        drop=0.1,
        name="cvae",
    ):

        super(CVAE, self).__init__()
        self.vocab_size = vocab
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.name = name

        self.embedding = nn.Embedding(self.vocab_size, emb_dim)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        self.dropout = nn.Dropout(p=drop)

        # self.c_encoder = nn.LSTM(emb_dim, hidden_size, bidirectional=True)
        self.x_encoder = nn.LSTM(
            emb_dim, hidden_size, bidirectional=True, batch_first=True
        )
        self.recognition = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.mu = nn.Linear(hidden_size * 2, latent_dim)
        self.log_var = nn.Linear(hidden_size * 2, latent_dim)

        # Make this latent + hidden (?)
        self.latent2hidden = nn.Linear(latent_dim, hidden_size)
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, self.vocab_size)

    def encode(self, x_emb, x_length, c):  # Produce Q(z | x, c)
        """
        x: (seq_len, batch_size)
        c: (batch_size, class_size (?))
        """
        # Cat x, c and encode
        # Embed x, c
        sorted_lengths, sorted_idx = torch.sort(x_length, descending=True)
        x_emb = x_emb[sorted_idx]

        # c_h, (c_hn, c_cn) = self.c_encoder(c)

        # Turn x to (seq_len, batch_size, emb_dim)
        packed_x = pack_padded_sequence(
            x_emb, sorted_lengths.data.tolist(), batch_first=True
        )

        x_h, (x_hn, x_cn) = self.x_encoder(packed_x)

        x_enc = torch.cat([x_hn[0], x_hn[1]], dim=-1)
        # c_enc = torch.cat([c_hn[0], c_hn[1]])
        # hidden_in = torch.cat([x_enc, c_enc], dim=-1)
        hidden_in = x_enc
        out_rec = F.elu(self.recognition(hidden_in))
        mu, log_var = self.mu(out_rec), self.log_var(out_rec)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        """
        Apply reparameterization for derivatives -> use rsample()?
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, x, x_lengths, z, c):  # Produce P(x | z, c)
        """
        z: (batch_size, latent_size (?))
        c: (batch_size, class_size (?))
        """
        # to_decode = torch.cat([z, c], dim=-1)
        to_decode = z

        x_lengths = [x + 1 for x in x_lengths]
        sorted_lengths, sorted_idx = torch.sort(x_lengths, descending=True)
        x = x[sorted_idx]
        packed_x = pack_padded_sequence(
            x, sorted_lengths.data.tolist(), batch_first=True
        )

        output, hidden = self.decoder(packed_x, to_decode)

        # Unpack and then return to original order
        padded_outputs = pad_packed_sequence(output, batch_first=True)[0]
        _, reversed_idx = torch.sort(sorted_idx)
        padded_outputs = padded_outputs[reversed_idx]

        # Project output to vocab
        output = F.softmax(self.out(padded_outputs), dim=-1)
        return output

    def forward(self, x, x_lengths, c):

        # Embed the padded input
        x_emb = self.dropout(self.embedding(x))

        mu, log_var = self.encode(x_emb, x_lengths, c)

        # Handle formatting the latent properly for LSTM
        z = self.reparameterize(mu, log_var)
        hidden = self.latent2hidden(z)
        hidden = (hidden.unsqueeze(0), hidden.unsqueeze(0))

        # Teacher forcing here - Preppend SOS token
        SOS = torch.ones(x.shape[0], 1).long().to(self.device())
        SOS = self.dropout(self.embedding(SOS))

        teacher_force = torch.stack([SOS, x_emb], dim=0)
        out_seq = self.decode(teacher_force, hidden, c)

        return out_seq, mu, log_var
