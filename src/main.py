"""
main.py

PURPOSE: This file defines the driving functions for the expirements/code
        of the project and contains the argparser
"""
import argparse
from train import train


def _parse_args():
    """
    Parses the commandline arguments for running an expirement trail/series
    of trials

    Args:

    Returns:
        args: the parsed arguments in a new namespace
    """

    """
    Add arguments below.  Example format:
        parser.add_argument('-cp', '--continue_training_policy',
            action='store_true', help='A help message'
        )

        parser.add_argument('--q1_checkpoint_filename', type=str,
            default='./q1_checkpoint.pth', help="Name of file to save and load"
        )
    """
    parser = argparse.ArgumentParser(
        description="Arguments for Lyric Generation project",
    )
    # Model related arguments here

    # Training related arguments here
    parser.add_argument("--rand_seed", type=int, default=1)
    parser.add_argument(
        "--data", type=str, default="./data/wikitext2", help="Dir of dataset"
    )
    parser.add_argument("--log_dir", type=str, default=None, help="Dir of tb")
    parser.add_argument("-n", "--num_epoch", type=int, default=50)
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3)
    parser.add_argument("-c", "--continue_training", action="store_true")
    parser.add_argument("-b", "--batch_size", type=int, default=2)
    parser.add_argument("-gc", "--grad_clip", type=float, default=0.0)

    args = parser.parse_args()
    return args


def main():
    args = _parse_args()
    train(args)


if __name__ == "__main__":
    main()
