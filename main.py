"""
MVCLEEA: Multi-View Contrastive Learning for Enzyme Function Annotation

Main entry point for training and evaluation.
"""
import argparse
import torch
from train import main as train_main


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='MVCLEEA: Multi-View Contrastive Learning for Enzyme Function Annotation'
    )
    subparsers = parser.add_subparsers(dest='command')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train the MVCLEEA model')
    train_parser.add_argument('--config', type=str, default='configs/default.yaml',
                              help='Path to config file')
    train_parser.add_argument('--device', type=str, default='cuda:0',
                              help='Device (e.g., cuda:0, cuda:1, cpu)')
    train_parser.add_argument('--run-name', type=str, default=None,
                              help='Name for this training run')

    # Test command
    test_parser = subparsers.add_parser('test', help='Run forward pass verification')
    test_parser.add_argument('--num-classes', type=int, default=588,
                             help='Number of EC classes')

    args = parser.parse_args()

    if args.command == 'train':
        train_main()
    elif args.command == 'test':
        from test_model import test_individual_modules, test_full_model, test_joint_loss
        print("Running MVCLEEA forward pass verification...\n")
        test_individual_modules()
        test_full_model()
        test_joint_loss()
        print("\nAll tests passed.")
    else:
        parser.print_help()
