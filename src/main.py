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
    # Hyperparameters are default from Song et al. 2019
    # Persona information for...
    parser.add_argument("-e", "--embedding", type=int, default=300)
    parser.add_argument("-hd", "--hidden", type=int, default=500)
    parser.add_argument("-l", "--latent", type=int, default=100)
    parser.add_argument("-d", "--dropout", type=float, default=0.1)
    parser.add_argument('--rnn', type=str, default='lstm')

    # Training related arguments here
    parser.add_argument("--rand_seed", type=int, default=1)
    parser.add_argument(
        "--data", type=str, default="./data/wikitext2", help="Dir of dataset"
    )
    parser.add_argument(
        "--persona_data", type=str, default="./data/personas.json", help="Dir of dataset"
    )
    parser.add_argument("--log_dir", type=str, default=None, help="Dir of tb")
    parser.add_argument("-n", "--num_epoch", type=int, default=50)
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3)
    parser.add_argument("-c", "--continue_training", action="store_true")
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument("-gc", "--grad_clip", type=float, default=0.0)

    args = parser.parse_args()
    return args


def main():
    args = _parse_args()
    train(args)


if __name__ == "__main__":
    main()
