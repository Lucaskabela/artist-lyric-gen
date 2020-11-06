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
        self.tanh = torch.nn.Tanh()

        # X and P will be encoded then concatenated
        self.x_encoder = nn.LSTM(emb_dim, hidden_size, bidirectional=True, batch_first=True)
        self.p_encoder = nn.LSTM(emb_dim, hidden_size, bidirectional=True, batch_first=True)

        self.y_encoder = nn.LSTM(
            emb_dim, hidden_size, bidirectional=True, batch_first=True
        )

        self.recognition = nn.Linear(hidden_size * 6, hidden_size * 2)
        self.r_mu_log_var = nn.Linear(hidden_size * 2, latent_dim * 2)

        self.prior = nn.Linear(hidden_size * 4, hidden_size * 2)
        self.p_mu_log_var = nn.Linear(hidden_size * 2, latent_dim * 2)

        # Make this latent + hidden (?)
        self.latent2hidden = nn.Linear(latent_dim + hidden_size * 4, hidden_size)
        self.decoder = nn.LSTM(emb_dim,  hidden_size, batch_first=True)
        self.out = nn.Linear( hidden_size, self.vocab_size)

    def encode(self, x_emb, x_length, p_emb, p_length, y_emb, y_length):  # Produce Q(z | x, y, p)
        """
        x: (seq_len, batch_size)
        c: (batch_size, class_size (?))
        """
        # For efficency, sort and pack x, then encode!
        sorted_x_lengths, sorted_x_idx = torch.sort(x_length, descending=True)
        x_emb = x_emb[sorted_x_idx]
        packed_x = pack_padded_sequence(
            x_emb, sorted_x_lengths.data.tolist(), batch_first=True
        )
        x_h, (x_hn, x_cn) = self.x_encoder(packed_x)
        x_enc = torch.cat([x_hn[0], x_hn[1]], dim=-1)

        # For efficency, sort and pack p, then encode!
        sorted_p_lengths, sorted_p_idx = torch.sort(p_length, descending=True)
        p_emb = p_emb[sorted_p_idx]
        packed_p = pack_padded_sequence(
            p_emb, sorted_p_lengths.data.tolist(), batch_first=True
        )
        p_h, (p_hn, p_cn) = self.p_encoder(packed_p)
        p_enc = torch.cat([p_hn[0], p_hn[1]], dim=-1)


        # For efficency, sort and pack y, then encode!
        sorted_y_lengths, sorted_y_idx = torch.sort(y_length, descending=True)
        y_emb = y_emb[sorted_y_idx]
        packed_y = pack_padded_sequence(
            y_emb, sorted_y_lengths.data.tolist(), batch_first=True
        )
        y_h, (y_hn, y_cn) = self.y_encoder(packed_y)
        y_enc = torch.cat([y_hn[0], y_hn[1]], dim=-1)

        c_enc = torch.cat([x_enc, p_enc], dim=-1)

        # Should I concatenate context here too?
        hidden_in = torch.cat([y_enc, c_enc], dim=-1)
        out_rec = self.tanh(self.recognition(hidden_in))
        out_prior = self.tanh(self.prior(c_enc))

        r_mu, r_log_var = torch.split(self.r_mu_log_var(out_rec), self.latent_dim, dim=-1)
        p_mu, p_log_var = torch.split(self.p_mu_log_var(out_prior), self.latent_dim, dim=-1)
        return r_mu, r_log_var, p_mu, p_log_var, c_enc

    def reparameterize(self, mu, log_var):
        """
        Apply reparameterization for derivatives -> use rsample()?
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, y, y_lens, to_decode):  # Produce P(x | z, c)
        """
        z: (batch_size, latent_size (?))
        c: (batch_size, class_size (?))
        """

        y_lengths = torch.LongTensor([y + 1 for y in y_lens]).to(self.device())
        sorted_lengths, sorted_idx = torch.sort(y_lengths, descending=True)
        y = y[sorted_idx]
        packed_y = pack_padded_sequence(
            y, sorted_lengths.data.tolist(), batch_first=True
        )

        output, hidden = self.decoder(packed_y, to_decode)

        # Unpack and then return to original order
        padded_outputs = pad_packed_sequence(output, batch_first=True)[0]
        _, reversed_idx = torch.sort(sorted_idx)
        padded_outputs = padded_outputs[reversed_idx]

        # Project output to vocab
        output = F.log_softmax(self.out(padded_outputs), dim=-1)
        return output

    def forward(self, x, x_lengths, p, p_lengths, y, y_lengths):

        # Embed the padded input
        x_emb = self.dropout(self.embedding(x))
        p_emb = self.dropout(self.embedding(p))
        y_emb = self.dropout(self.embedding(y))
        params = self.encode(x_emb, x_lengths, p_emb, p_lengths, y_emb, y_lengths)
        r_mu, r_log_var, p_mu, p_log_var, c = params

        # Handle formatting the latent properly for LSTM
        z = self.reparameterize(r_mu, r_log_var)
        to_decode = torch.cat([z, c], dim=-1).unsqueeze(0)
        hidden = self.latent2hidden(to_decode)
        to_decode = (hidden, hidden)

        # Teacher forcing here - Preppend SOS token
        SOS = torch.ones(y.shape[0], 1).long().to(self.device())
        SOS = self.dropout(self.embedding(SOS))

        teacher_force = torch.cat([SOS, y_emb], dim=1)
        out_seq = self.decode(teacher_force, y_lengths, to_decode)

        return out_seq, r_mu, r_log_var, p_mu, p_log_var
