"""
train.py

PURPOSE: This file defines the code for training the neural networks in pytorch
"""

from os import path
import models
import utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard as tb


def vae_loss_function(x_p, x, mu, log_var, alpha=0):
    """
    Loss for CVAE is BCE + KLD
        see Appendix B from Kingma and Welling 2014
    Need alpha for KL annealing
    """
    BCE = F.nll_loss(x_p, x, reduction="sum", ignore_index=0)
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + alpha * KLD


def seed_random(rand_seed):
    torch.manual_seed(rand_seed)
    np.random.seed(rand_seed)


def init_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def init_logger(log_dir=None):
    train_logger, valid_logger = None, None
    if log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(log_dir, "train"))
        valid_logger = tb.SummaryWriter(path.join(log_dir, "valid"))
    return train_logger, valid_logger


def eval_inference(model, corpus, valid_log, global_step, n=4):
    max_length = 30
    model.eval()
    z = torch.randn([n, model.latent_dim])
    z = model.latent2hidden(z)
    hidden = (z.unsqueeze(0), z.unsqueeze(0))

    # Teacher forcing here
    # required for dynamic stopping of sentence generation
    sequence_idx = torch.arange(0, n).long()  # all idx of batch
    # all idx of batch which are still generating
    sequence_running = torch.arange(0, n).long()
    sequence_mask = torch.ones(n).bool()
    # idx of still generating sequences with respect to current loop
    running_seqs = torch.arange(0, n).long()

    generations = torch.tensor(n, max_length).fill_(1).long()
    t = 0
    while t < max_length and len(running_seqs) > 0:

        if t == 0:
            input_sequence = torch.Tensor(n).fill_(1).long()

        input_sequence = input_sequence.unsqueeze(1)

        input_embedding = model.embedding(input_sequence)

        output, hidden = model.decode(input_embedding, hidden)

        logits = F.softmax(model.out(output), dim=-1)

        input_sequence = torch.max(logits, dim=-1)

        # save next input
        generations[running_seqs][t] = input_sequence

        # update gloabl running sequence
        sequence_mask[sequence_running] = input_sequence != 2
        sequence_running = sequence_idx.masked_select(sequence_mask)

        # update local running sequences
        running_mask = (input_sequence != 2).data
        running_seqs = running_seqs.masked_select(running_mask)

        # prune input and hidden state according to local update
        if len(running_seqs) > 0:
            input_sequence = input_sequence[running_seqs]
            hidden = hidden[:, running_seqs]

            running_seqs = torch.arange(0, len(running_seqs)).long()

        t += 1

    # Produce 4 examples here
    if valid_log is not None:
        for i in range(len(generations)):
            name_ = "generated_example_{}".format(i)
            valid_log.add_text(name_, str(generations[i]), global_step)
    else:
        for i in range(len(generations)):
            name_ = "generated_example_{}".format(i)
            print(name_, generations[i])


def train(args):
    """
    trains a model as specified by args
    """
    seed_random(args.rand_seed)
    device = init_device()
    train_log, valid_log = init_logger(log_dir=args.log_dir)

    # TODO: set up load_data functions - be best if return a data loader
    corpus = utils.Corpus(args.data)
    # This should return a dataloader or something to that effect
    train_data = utils.load_data(corpus.train, batch_size=args.batch_size)
    # This should return a dataloader or something to that effect

    vocab = len(corpus.dictionary)
    model = models.CVAE(vocab, args.embedding, args.hidden, args.latent)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    if args.continue_training:
        model.load_model()
    model = model.to(device)

    # TODO: Change the ignore_index to padding index
    loss = vae_loss_function

    global_step = 0
    for epoch in range(args.num_epoch):

        model.train()
        losses = []
        for x, x_len in train_data:
            # Now we need to make sure everything in the batch has same size
            x, x_len = x.to(device), x_len.to(device)
            pred, mu, log_var = model(x, x_len, None)
            eos_tensor = torch.empty(x.shape[0], 1).to(device)
            eos_tensor.fill_(corpus.dictionary.word2idx["<EOS>"])
            gold = torch.cat([x, eos_tensor], dim=1).long()
            alph = min(max(0, (global_step - 10_000) / 60_000), 1)
            pred = pred.permute(1, 0, 2)
            # Get loss, normalized by batch size
            loss_val = loss(pred, gold, mu, log_var, alpha=alph)
            loss_val /= args.batch_size

            optimizer.zero_grad()
            loss_val.backward()
            if args.grad_clip > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            global_step += 1

            losses.append(loss_val.detach().cpu().numpy())
            if train_log is not None:
                train_log.add_scalar("loss", losses[-1], global_step)

        eval_inference(model, corpus, valid_log, global_step)
        avg_l = np.mean(losses)
        print("epoch %-3d \t loss = %0.3f \t" % (epoch, avg_l))
        model.save_model()
    model.save_model()
