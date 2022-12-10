import cv2
import numpy as np

from orientation import keypoint_info, skeleton_info
from typing import List


def get_thickness(height: int, width: int):
    mx = max(height, width)
    if mx <= 250:
        return 3
    elif mx <= 1000:
        return 5
    return 10


def get_optimal_font_scale(text: str, width: int, thickness: int):
    for scale in reversed(range(0, 60, 1)):
        textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
            fontScale=scale/10, thickness=thickness)
        new_width = textSize[0][0]
        if (new_width <= width):
            return scale / 10
    return 1


def visualize_keypoints(image: np.ndarray, keypoints: List[float]):
    image = image.copy()
    keypoints = list(map(int, keypoints))

    def get_kth(k):
        return keypoints[k * 3], keypoints[k * 3 + 1]

    h, w, _ = image.shape
    thickness = get_thickness(h, w)
    for k, v in keypoint_info.items():
        image = cv2.circle(
            img=image, center=get_kth(k), radius=thickness,
            color=v["color"], thickness=-1)
    for k, v in skeleton_info.items():
        image = cv2.line(
            img=image, pt1=get_kth(v["link"][0]), pt2=get_kth(v["link"][1]),
            color=v["color"], thickness=thickness
        )

    return image


# map for visualizing orientation
# contains: (name, bgcolor, fgcolor)
orientation_map = [
    ("front", (32, 165, 218), (255, 255, 255)),
    ("lfront", (35, 142, 107), (255, 255, 255)),
    ("rfront", (143, 188, 143), (255, 255, 255)),
    ("left", (235, 206, 135), (255, 255, 255)),
    ("right", (237, 149, 100), (255, 255, 255)),
    ("back", (238, 130, 238), (255, 255, 255)),
    ("unknown", (0, 0, 0), (255, 255, 255)),
]


def mark_orientation(image: np.ndarray, orientation: int):
    image = image.copy()
    h, w, c = image.shape
    ratio = 8
    name, bg, fg = orientation_map[orientation]
    content = f"{orientation}-{name}"
    thickness = get_thickness(h, w)
    scale = get_optimal_font_scale(content, int(w * 0.95), thickness)
    text_width, text_height = cv2.getTextSize(content, 
        fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
        fontScale=scale, thickness=thickness)[0]
    label = np.zeros((h // ratio, w, c), dtype=image.dtype)
    label[:, :] = bg
    label = cv2.putText(
        img=label, text=content, 
        org=(w // 2 - text_width // 2, h // ratio // 2 + text_height // 2),
        fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=scale,
        color=fg, thickness=thickness * 2 // 3)
    return np.concatenate((image, label), axis=0)
