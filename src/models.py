"""
models.py

PURPOSE: This file defines Neural Network Architecture and other models
        which will be evaluated in this expirement
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.name = name

        self.embedding = nn.Embedding(self.vocab_size, emb_dim)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        self.dropout = nn.Dropout(p=drop)

        # self.c_encoder = nn.LSTM(emb_dim, hidden_size, bidirectional=True)
        self.x_encoder = nn.LSTM(emb_dim, hidden_size, bidirectional=True)
        self.recognition = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.mu = nn.Linear(hidden_size * 2, latent_dim)
        self.log_var = nn.Linear(hidden_size * 2, latent_dim)

        # Make this latent + hidden (?)
        self.decoder_embedding = nn.Embedding(self.vocab_size, latent_dim)
        self.decoder = nn.LSTM(latent_dim, hidden_size)
        self.out = nn.Linear(hidden_size, self.vocab_size)

    def encode(self, x, c):  # Produce Q(z | x, c)
        """
        x: (seq_len, batch_size, embedding_dim)
        c: (batch_size, class_size (?))
        """
        # Cat x, c and encode
        # Embed x, c

        # c_h, (c_hn, c_cn) = self.c_encoder(c)
        x = self.dropout(self.embedding(x))
        x_h, (x_hn, x_cn) = self.x_encoder(x)
        x_enc = torch.cat([x_hn[0], x_hn[1]])
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

    def decode(self, input_, z, c):  # Produce P(x | z, c)
        """
        z: (batch_size, latent_size (?))
        c: (batch_size, class_size (?))
        """
        # to_decode = torch.cat([z, c], dim=-1)
        to_decode = z
        embedded_prev = self.decoder_embedding(input_)
        embedded_prev = self.dropout(embedded_prev)

        output, hidden = self.decoder(embedded_prev, to_decode)
        output = F.softmax(self.out(output[0]))
        return output, hidden

    def forward(self, x, c):
        mu, log_var = self.encode(x, c)
        z = self.reparameterize(mu, log_var)

        # Teacher forcing here
        SOS = torch.ones(x.shape[0], 1, 1).long().to(self.device())
        initial = self.decode(SOS, z, c)
        out_sequence = [initial]
        for token in x:
            token_t, hidden = self.decode(token, z, c)
            out_sequence.append(token_t)
            z = hidden

        out_seq = torch.stack(out_sequence)
        return out_seq, mu, log_var
