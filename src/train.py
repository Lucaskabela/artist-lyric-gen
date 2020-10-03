"""
train.py

PURPOSE: This file defines the code for training the neural networks in pytorch
"""

from .models import save_model, load_model
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.tensorboard as tb
import torchvision.transforms.functional as TF


def train(args):
    from os import path

    model = None  # Planner()
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
        model = load_model(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_data = load_data("drive_data")
    loss = torch.L1Loss()
    global_step = 0
    for epoch in range(args.num_epoch):

        model.train()
        losses = []
        for img, label in train_data:
            img, label = img.to(device), label.to(device)

            pred = model(img)
            loss_val = loss(pred, label)

            if train_logger is not None:
                train_logger.add_scalar("loss", loss_val, global_step)
                if global_step % 100 == 0:
                    fig, ax = plt.subplots(1, 1)
                    ax.imshow(TF.to_pil_image(img[0].cpu()))
                    ax.add_artist(plt.Circle(label[0], 2, ec="g", fill=False, lw=1.5))
                    ax.add_artist(plt.Circle(pred[0], 2, ec="r", fill=False, lw=1.5))
                    train_logger.add_figure("viz", fig, global_step)
                    del ax, fig

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1

            losses.append(loss_val.detach().cpu().numpy())

        avg_loss = np.mean(losses)
        if train_logger is None:
            print("epoch %-3d \t loss = %0.3f" % (epoch, avg_loss))
        save_model(model)

    save_model(model)
