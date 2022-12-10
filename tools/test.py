import argparse
import cv2
import os
import sys
import json

sys.path.append(".")
import pickle
import numpy as np
from pprint import pprint
from tqdm import tqdm
from datetime import datetime
from orientation.dataset import Dataset
from orientation.metrics import get_metrics
from orientation.visualize import visualize_keypoints, mark_orientation


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, choices=[
        "rb",  # rule-based
        "mlp",  # ml-based MlpClassifier
    ], default='mlp')
    parser.add_argument("--subset", type=str, choices=[
        "all", "train", "test"
    ], default="test")
    parser.add_argument("--work_dir", type=str,
                        default='/Users/longruihan/代码/orientation/tools/workdir/20221210002005')
    parser.add_argument("--visualize", default=True, action="store_true")
    return parser.parse_args()


def get_hook(args):
    if args.method == "rb":
        from orientation.parse.rule_based import test_dataset
        return test_dataset
    elif args.method == "mlp":
        from orientation.parse.ml_based import MlpClassifier
        mlp = MlpClassifier(args)
        return mlp.test
    raise NotImplementedError(args.method)


def get_dataset(args):
    return Dataset(args.subset)


def visualize(dataset, pred, args):
    vis_dir = os.path.join(args.work_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)
    for idx, (image, info) in enumerate(tqdm(dataset)):
        img_vis = visualize_keypoints(image, info["pose"]["keypoints"])
        img_vis = mark_orientation(img_vis, pred[idx])
        img_path = os.path.join(vis_dir, info["filename"])
        cv2.imwrite(img_path, img_vis)


label2view = {0: 'front_view', 1: 'left_view', 2: 'right_view', 3: 'left_view', 4: 'right_view', 5: 'back_view'}



def main():
    args = get_args()
    print(args)
    hook = get_hook(args)
    dataset = get_dataset(args)
    # pred = hook(dataset)
    pred,filename = hook(dataset)
    print(len(pred))
    print(len(filename))
    #view_dict = {}
    # for i in range(len(filename)):
    #     view_dict[filename[i].split('.')[0]] = label2view[pred[i]]
    # with open('/Users/longruihan/课题组/orientation/dml_seven_v2_query_view_2.pickle', 'wb') as f:
    #     view_dict = pickle.dumps(view_dict)
    #     f.write(view_dict)

    gt = [info["orientation"] for _, info in dataset]
    if args.visualize:
        visualize(dataset, pred, args)
    pprint(get_metrics(np.array(pred), np.array(gt)))


if __name__ == "__main__":
    main()
