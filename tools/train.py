import argparse
import os
import sys
sys.path.append(".")
from datetime import datetime
from orientation.dataset import Dataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str,default='mlp', choices=[
        "mlp", # ml-based MlpClassifier
    ])
    parser.add_argument("--subset", type=str, choices=[
        "all", "train", "test"
    ], default="train")
    parser.add_argument("--work_dir", type=str,
        default=os.path.join("workdir", datetime.now().strftime(r"%Y%m%d%H%M%S")))
    return parser.parse_args()


def get_hook(args):
    if args.method == "mlp":
        from orientation.parse.ml_based import MlpClassifier
        mlp = MlpClassifier(args)
        return mlp.train
    raise NotImplementedError(args.method)


def get_dataset(args):
    return Dataset(args.subset)


def main():
    args = get_args()
    print(args)
    hook = get_hook(args)
    os.makedirs(args.work_dir, exist_ok=True)
    dataset = get_dataset(args)
    hook(dataset)


if __name__ == "__main__":
    main()
