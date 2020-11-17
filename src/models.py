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
import random


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

    def num_params(self):
        pytorch_total_params = sum(p.numel() for p in self.parameters())
        print(pytorch_total_params)

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
        rnn="lstm"
    ):

        super(CVAE, self).__init__()
        self.vocab_size = vocab
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.name = name
        self.rnn = rnn

        self.embedding = nn.Embedding(self.vocab_size, emb_dim)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)

        # Create the dropout layers
        self.emb_dropout = nn.Dropout(p=drop)
        self.latent_dropout = nn.Dropout(p=drop)
        self.hidden_dropout = nn.Dropout(p=drop)

        # Layer Norms for stability
        self.recoglnorm = nn.LayerNorm(hidden_size * 2)
        self.priorlnorm = nn.LayerNorm(hidden_size * 2)

        # Because F.... is depricated
        self.tanh = torch.tanh

        # X and P will be encoded then concatenated
        if rnn == "lstm":
            self.x_encoder = nn.LSTM(emb_dim, hidden_size, bidirectional=True, batch_first=True)
            self.p_encoder = nn.LSTM(emb_dim, hidden_size, bidirectional=True, batch_first=True)

            self.y_encoder = nn.LSTM(
                emb_dim, hidden_size, bidirectional=True, batch_first=True
            )
        elif rnn == "gru":
            self.x_encoder = nn.GRU(emb_dim, hidden_size, bidirectional=True, batch_first=True)
            self.p_encoder = nn.GRU(emb_dim, hidden_size, bidirectional=True, batch_first=True)

            self.y_encoder = nn.GRU(
                emb_dim, hidden_size, bidirectional=True, batch_first=True
            )

        self.recognition = nn.Linear(hidden_size * 6, hidden_size * 2)
        self.r_mu_log_var = nn.Linear(hidden_size * 2, latent_dim * 2)

        self.prior = nn.Linear(hidden_size * 4, hidden_size * 2)
        self.p_mu_log_var = nn.Linear(hidden_size * 2, latent_dim * 2)

        # Make this latent + hidden (?)
        self.latent2hidden = nn.Linear(latent_dim + hidden_size * 4, hidden_size)
        self.bow1 = nn.Linear(latent_dim + hidden_size * 4, hidden_size)
        self.bow2 = nn.Linear(hidden_size, self.vocab_size)

        if rnn == "lstm":
            self.decoder = nn.LSTM(emb_dim,  hidden_size, batch_first=True)
        elif rnn == "gru":
            self.decoder = nn.GRU(emb_dim,  hidden_size, batch_first=True)
        else:
            raise Exception("RNN type {} is not supported".format(rnn))

        self.out = nn.Linear( hidden_size, self.vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def contextualize(self, x_emb, x_length, p_emb, p_length):
        # For efficency, sort and pack x, then encode!
        sorted_x_lengths, sorted_x_idx = torch.sort(x_length, descending=True)
        x_emb = x_emb[sorted_x_idx]
        packed_x = pack_padded_sequence(
            x_emb, sorted_x_lengths.data.tolist(), batch_first=True
        )
        x_h, x_hn = self.x_encoder(packed_x)
        if self.rnn == "lstm":
            # Disregard the cell if lstm
            x_hn = x_hn[0]
        x_enc = torch.cat([x_hn[0], x_hn[1]], dim=-1)

        # For efficency, sort and pack p, then encode!
        sorted_p_lengths, sorted_p_idx = torch.sort(p_length, descending=True)
        p_emb = p_emb[sorted_p_idx]
        packed_p = pack_padded_sequence(
            p_emb, sorted_p_lengths.data.tolist(), batch_first=True
        )
        p_h, p_hn = self.p_encoder(packed_p)
        if self.rnn == "lstm":
            # Disregard the cell if lstm
            p_hn = p_hn[0]

        p_enc = torch.cat([p_hn[0], p_hn[1]], dim=-1)
        c_enc = torch.cat([x_enc, p_enc], dim=-1)
        return c_enc

    def encode(self, x_emb, x_length, p_emb, p_length, y_emb, y_length):  # Produce Q(z | x, y, p)
        """
        x: (seq_len, batch_size)
        c: (batch_size, class_size (?))
        """
        c_enc = self.contextualize(x_emb, x_length, y_emb, y_length)

        # For efficency, sort and pack y, then encode!
        sorted_y_lengths, sorted_y_idx = torch.sort(y_length, descending=True)
        y_emb = y_emb[sorted_y_idx]
        packed_y = pack_padded_sequence(
            y_emb, sorted_y_lengths.data.tolist(), batch_first=True
        )
        y_h, y_hn = self.y_encoder(packed_y)
        if self.rnn == "lstm":
            # Disregard the cell if lstm
            y_hn = y_hn[0]
        y_enc = torch.cat([y_hn[0], y_hn[1]], dim=-1)

        # Should I concatenate context here too?
        hidden_in = torch.cat([y_enc, c_enc], dim=-1)
        out_rec = self.recoglnorm(self.tanh(self.recognition(hidden_in)))
        out_prior = self.priorlnorm(self.tanh(self.prior(c_enc)))

        r = self.latent_dropout(self.r_mu_log_var(out_rec))
        r_mu, r_log_var = torch.split(r, self.latent_dim, dim=-1)

        p = self.latent_dropout(self.p_mu_log_var(out_prior))
        p_mu, p_log_var = torch.split(p, self.latent_dim, dim=-1)

        return r_mu, r_log_var, p_mu, p_log_var, c_enc

    def reparameterize(self, mu, log_var):
        """
        Apply reparameterization for derivatives -> use rsample()?
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, y, y_lens, to_decode, teacher_ratio=1):  # Produce P(x | z, c)
        """
        z: (batch_size, latent_size (?))
        c: (batch_size, class_size (?))
        """
        use_teacher = random.random() < teacher_ratio
        y_lengths = torch.LongTensor([y + 1 for y in y_lens]).to(self.device())
        sorted_lengths, sorted_idx = torch.sort(y_lengths, descending=True)

        if use_teacher:
            y = y[sorted_idx]
            packed_y = pack_padded_sequence(
                y, sorted_lengths.data.tolist(), batch_first=True
            )
            output, hidden = self.decoder(packed_y, to_decode)
            # Unpack and then return to original order
            padded_outputs = pad_packed_sequence(output, batch_first=True)[0]
            _, reversed_idx = torch.sort(sorted_idx)
            padded_outputs = padded_outputs[reversed_idx]
        else:
            decoder_outputs = []
            # Get SOS as input
            decoder_input = y[:, 0, :].unsqueeze(1)
            hidden = to_decode
            # Go for length of longest sequence
            for di in range(sorted_lengths[0]):
                output, hidden = self.decoder(decoder_input, hidden)
                topv, topi = output.topk(1)
                decoder_input = topi.squeeze(1).detach()  # detach from history as input
                decoder_outputs.append(output)
                decoder_input = self.emb_dropout(self.embedding(decoder_input))
            padded_outputs = torch.stack(decoder_outputs, dim=1).squeeze()

        # Project output to vocab
        output = self.log_softmax(self.out(padded_outputs))
        return output

    def bow_logits(self, to_decode, max_len):
        res = self.bow2(self.hidden_dropout(torch.tanh(self.bow1(to_decode))))
        res = res.squeeze(0).unsqueeze(1)
        return torch.repeat_interleave(res, max_len, dim=1)

    def forward(self, x, x_lengths, p, p_lengths, y, y_lengths, teacher_ratio=1):

        # Embed the padded input
        x_emb = self.emb_dropout(self.embedding(x))
        p_emb = self.emb_dropout(self.embedding(p))
        y_emb = self.emb_dropout(self.embedding(y))
        params = self.encode(x_emb, x_lengths, p_emb, p_lengths, y_emb, y_lengths)
        r_mu, r_log_var, p_mu, p_log_var, c = params

        # Handle formatting the latent properly for LSTM
        z = self.reparameterize(r_mu, r_log_var)
        to_decode = torch.cat([z, c], dim=-1).unsqueeze(0)

        bow_log = self.bow_logits(to_decode, max(y_lengths))

        hidden = self.hidden_dropout(self.latent2hidden(to_decode))
        if self.rnn == "lstm":
            to_decode = (hidden, hidden)
        else:
            to_decode = hidden

        # Teacher forcing here - Preppend SOS token
        SOS = torch.ones(y.shape[0], 1).long().to(self.device())
        SOS = self.emb_dropout(self.embedding(SOS))

        teacher_force = torch.cat([SOS, y_emb], dim=1)
        out_seq = self.decode(teacher_force, y_lengths, to_decode, teacher_ratio=teacher_ratio)

        return out_seq, bow_log, r_mu, r_log_var, p_mu, p_log_var

    def infer_hidden(self, x, x_lengths, p, p_lengths):
        # Embed the padded input
        x_emb = self.emb_dropout(self.embedding(x))
        p_emb = self.emb_dropout(self.embedding(p))

        c_enc = self.contextualize(x_emb, x_lengths, p_emb, p_lengths)
        out_prior = self.priorlnorm(self.tanh(self.prior(c_enc)))

        p = self.latent_dropout(self.p_mu_log_var(out_prior))
        p_mu, p_log_var = torch.split(p, self.latent_dim, dim=-1)

        z = self.reparameterize(p_mu, p_log_var)
        to_decode = torch.cat([z, c_enc], dim=-1).unsqueeze(0)
        hidden = self.latent2hidden(to_decode)

        return (hidden, hidden) if self.rnn == "lstm" else hidden

class VAE(BaseNetwork):
    """
    Defines the VAE approach, using RNNs as Encoder/Decoder
    """

    def __init__(
        self,
        vocab,
        emb_dim,
        hidden_size,
        latent_dim,
        drop=0.1,
        rnn="lstm",
        name="vae",
    ):

        super(VAE, self).__init__()
        self.vocab_size = vocab
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.name = name
        self.rnn = rnn

        self.embedding = nn.Embedding(self.vocab_size, emb_dim)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        self.dropout = nn.Dropout(p=drop)
        self.recoglnorm = nn.LayerNorm(hidden_size * 2)

        # X and P will be encoded then concatenated
        if rnn == "lstm":
            self.x_encoder = nn.LSTM(emb_dim, hidden_size, bidirectional=True, batch_first=True)
        elif rnn == "gru":
            self.x_encoder = nn.GRU(emb_dim, hidden_size, bidirectional=True, batch_first=True)

        self.recognition = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.r_mu_log_var = nn.Linear(hidden_size * 2, latent_dim * 2)

        # Make this latent + hidden (?)
        self.latent2hidden = nn.Linear(latent_dim, hidden_size)

        if rnn == "lstm":
            self.decoder = nn.LSTM(emb_dim,  hidden_size, batch_first=True)
        elif rnn == "gru":
            self.decoder = nn.GRU(emb_dim,  hidden_size, batch_first=True)
        else:
            raise Exception("RNN type {} is not supported".format(rnn))

        self.out = nn.Linear( hidden_size, self.vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def encode(self, x_emb, x_length):  # Produce Q(z | x, c)
        """
        x: (seq_len, batch_size)
        c: (batch_size, class_size (?))
        """
        sorted_lengths, sorted_idx = torch.sort(x_length, descending=True)
        x_emb = x_emb[sorted_idx]
        packed_x = pack_padded_sequence(
            x_emb, sorted_lengths.data.tolist(), batch_first=True
        )
        x_h, (x_hn, x_cn) = self.x_encoder(packed_x)

        x_enc = torch.cat([x_hn[0], x_hn[1]], dim=-1)

        out_rec = torch.tanh(self.recognition(x_enc))
        r = self.latent_dropout(self.r_mu_log_var(out_rec))
        r_mu, r_log_var = torch.split(r, self.latent_dim, dim=-1)
        return r_mu, r_log_var

    def reparameterize(self, mu, log_var):
        """
        Apply reparameterization for derivatives -> use rsample()?
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, x, x_lens, z):  # Produce P(x | z, c)
        """
        z: (batch_size, latent_size (?))
        c: (batch_size, class_size (?))
        """
        to_decode = z
        x_lengths = torch.LongTensor([x + 1 for x in x_lens]).to(self.device())
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
        output = self.log_softmax(self.out(padded_outputs), dim=-1)
        return output

    def forward(self, x, x_lengths):

        # Embed the padded input
        x_emb = self.dropout(self.embedding(x))

        mu, log_var = self.encode(x_emb, x_lengths)

        # Handle formatting the latent properly for LSTM
        z = self.reparameterize(mu, log_var)
        hidden = self.latent2hidden(z)
        if self.rnn == "lstm":
            to_decode = (hidden.unsqueeze(0), hidden.unsqueeze(0))
        else:
            to_decode = hidden

        # Teacher forcing here - Preppend SOS token
        SOS = torch.ones(x.shape[0], 1).long().to(self.device())
        SOS = self.dropout(self.embedding(SOS))

        teacher_force = torch.cat([SOS, x_emb], dim=1)
        out_seq = self.decode(teacher_force, x_lengths, hidden)

        return out_seq, mu, log_var