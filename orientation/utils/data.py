import numpy as np
from orientation.dataset import Dataset
from .keypoint import normalize as norm


def to_ml_format(dataset: Dataset, normalize=True):
    x, y, z = [], [], []
    for _, info in dataset:
        try:
            norm_func = norm if normalize else lambda x: x
            x.append(norm_func(info["pose"]["keypoints"]))
            y.append(info["orientation"])
            z.append(info["filename"])
        except:
            # print(info['filename'])
            continue
    return np.array(x), np.array(y), z
