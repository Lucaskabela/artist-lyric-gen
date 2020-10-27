"""
train.py

PURPOSE: This file defines the code for training the neural networks in pytorch
"""

from os import path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard as tb


def cvae_loss_function(x_p, x, mu, log_var):
    """
    Loss for CVAE is BCE + KLD
        see Appendix B from Kingma and Welling 2014
    """
    BCE = F.binary_cross_entropy(x_p, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD


def train(args):
    """
    trains a model as specified by args
    """
    model = None  # LSTM() or whatever - should be specified in args
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, "train"))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, "valid"))

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    if args.continue_training:
        model.load_model()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # TODO: set up load_data functions - be best if return a data loader
    train_data = None  # load_data()
    valid_data = None  # load_data()

    # TODO: Change the ignore_index to padding index
    loss = nn.CrossEntropyLoss(ignore_index=-1)

    global_step = 0
    for epoch in range(args.num_epoch):

        model.train()
        losses = []
        for x, y in train_data:
            x, y = x.to(device), y.to(device)

            # TODO: Add teacher forcing - need to pass actual
            pred = model(x)
            loss_val = loss(pred, y)

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            global_step += 1

            losses.append(loss_val.detach().cpu().numpy())
            if train_logger is not None:
                train_logger.add_scalar("loss", losses[-1], global_step)

        model.eval()
        eval_metrics = []
        for x, y in valid_data:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            # Compute whatever metrics here
            eval_metrics.append(0)

        avg_eval_metric = np.mean(eval_metrics)
        if valid_logger is not None:
            valid_logger.add_scalar("eval", avg_eval_metric, global_step)

        f1 = avg_eval_metric
        avg_l = np.mean(losses)
        print("epoch %-3d \t loss = %0.3f \t f1 = %.3f" % (epoch, avg_l, f1))
        model.save_model()
    model.save_model()
