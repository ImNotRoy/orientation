import json
import os
import cv2

from orientation import pkg_path


class Dataset:

    def __init__(self, subset="all"):
        assert subset in ["all", "train", "test"]
        self.data_root = os.path.join(
            os.path.dirname(pkg_path), "data"
        )
        if subset == 'test':
            with open('/Users/longruihan/代码/orientation/data/test.json', "r") as fi:
                self.annotations = json.load(fi)
        elif subset == 'train':
            with open('/Users/longruihan/代码/orientation/data/train.json', "r") as fi:
                self.annotations = json.load(fi)
        else:
            with open(os.path.join(self.data_root, f"{subset}.json"), "r") as fi:
                self.annotations = json.load(fi)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        item = self.annotations[index]
        image = cv2.imread(os.path.join(
            self.data_root, "images", item["filename"]
        ))
        return image, item
