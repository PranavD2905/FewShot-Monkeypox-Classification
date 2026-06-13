import argparse
import yaml

from train import train
from evaluate import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train')
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.mode == 'train':
        train(cfg)
    else:
        evaluate(cfg, checkpoint_path=args.checkpoint)


if __name__ == '__main__':
    main()
