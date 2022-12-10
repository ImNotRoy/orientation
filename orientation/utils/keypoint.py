import numpy as np
from PIL import Image
from typing import List

openpose_kp_dir = {"nose": 0, "neck": 1, "right_shoulder": 2, "right_elbow": 3,
                   "right_wrist": 4, "left_shoulder": 5, "left_elbow": 6, "left_wrist": 7,
                   "right_hip": 8, "right_knee": 9, "right_ankle": 10, "left_hip": 11,
                   "left_knee": 12, "left_ankle": 13, "right_eye": 14, "left_eye": 15,
                   "right_ear": 16, "left_ear": 17}

upper_body_kp_dir = {"right_eye": 14, "left_eye": 15, "right_ear": 16, "left_ear": 17, "nose": 0, "right_shoulder": 2,
                     "left_shoulder": 5}
lower_body_kp_dir = {"right_ankle": 10, "left_ankle": 13, "right_knee": 9, "left_knee": 12, "right_hip": 8,
                     "left_hip": 11}


def select_main_character_based_distance(pose_data):
    '''
    detail:
        Select the main character from persons which were detected from the image.
        the code from keypoint_utils_from_yirui.py. Based on the code，mainly update the conditions that select the main character.
        Only select main character of min distance that between the people and center of the picture.
    input:
        pose_data: {'image_size': {'width': int, 'height': int},
                    'pose': [float []]}
    return:
        mc_pose_data: float list ,[]
    '''
    pose_list = pose_data['pose']
    image_size = pose_data['image_size']
    mid_p = (image_size['width'] / 2, image_size['height'] / 2)
    min_dist = float('inf')
    mc_pose_data = []
    for pose in pose_list:
        avg_p = [0, 0]
        counter = 0
        for i in range(18):
            if pose[i * 3 + 2]:
                avg_p[0] = avg_p[0] + pose[i * 3]
                avg_p[1] = avg_p[1] + pose[i * 3 + 1]
                counter = counter + 1
        avg_p = (avg_p[0] / counter, avg_p[1] / counter)
        dist = vector_distance(avg_p, mid_p)
        if dist < min_dist:
            min_dist = dist
            mc_pose_data = pose
    return mc_pose_data


def vector_distance(vec1, vec2):
    '''
    get distance between vec1 and arr2
    :param vec1: np array
    :param vec2: np array
    :return: scalar
    '''
    return np.linalg.norm(vec1 - vec2)


def select_main_character_based_size(pose_data_list):
    '''
    Select main character based on size of people in the picture. Sometimes, the select_main_character function from keypoint2angle.py select main character may be passerby.
    So I  re-write the function.
    :param pose_data_list: multi people pose data,dtype: list
    :return: one people pose data,dtype: list
    '''
    if len(pose_data_list) == 1:
        return pose_data_list[0]
    else:
        people_size = 0
        tmp_pose_data = []
        for pose in pose_data_list:  # traverse each pose data and its key-point from bottom and top that the confidence is not equal zero.
            upper_body_kp_idx = 0
            lower_body_kp_idx = 0
            for k, v in upper_body_kp_dir.items():  # TODO traverse method of python dict
                if pose[v * 3] != 0:
                    upper_body_kp_idx = v
                    break
            for k, v in lower_body_kp_dir.items():
                if pose[v * 3] != 0:
                    lower_body_kp_idx = v
                    break
            upper_body_kp_pos = np.array((pose[upper_body_kp_idx * 3], pose[upper_body_kp_idx * 3 + 1]))
            lower_body_kp_pos = np.array((pose[lower_body_kp_idx * 3], pose[lower_body_kp_idx * 3 + 1]))

            tmp_size = vector_distance(upper_body_kp_pos, lower_body_kp_pos)
            if tmp_size > people_size:
                people_size = tmp_size
                tmp_pose_data = pose
        return tmp_pose_data


def get_image_size(img_path: str):
    img = Image.open(img_path)
    w = img.width
    h = img.height
    return w, h


def normalize(keypoints: List[float]):
    """
    Normalize the keypoints coordinates along x, y and z axis
    :param keypoints: should be a list of float with length of 51
    :return: normalized keypoints list
    """
    r = keypoints[:]
    for start in range(3):
        indices = list(range(start, len(keypoints), 3))
        mx = max([keypoints[idx] for idx in indices])
        for idx in indices:
            r[idx] = keypoints[idx] / mx
    return r


if __name__ == "__main__":
    print(normalize([
        70.45117950439453,
        13.470703125,
        0.5546790361404419,
        35.00194549560547,
        6.380859375,
        0.3937569856643677,
        68.08789825439453,
        11.107421875,
        0.6580307483673096,
        37.36522674560547,
        8.744140625,
        0.8469634056091309,
        65.72461700439453,
        11.107421875,
        0.8533499240875244,
        25.54882049560547,
        34.740234375,
        0.8836402893066406,
        72.81446075439453,
        37.103515625,
        0.8866669535636902,
        16.09569549560547,
        72.552734375,
        0.9342221021652222,
        84.63086700439453,
        70.189453125,
        0.8933306932449341,
        30.27538299560547,
        79.642578125,
        0.5920853614807129,
        65.72461700439453,
        67.826171875,
        0.4807667136192322,
        32.63866424560547,
        103.275390625,
        0.7652077078819275,
        60.99805450439453,
        105.638671875,
        0.7596336007118225,
        35.00194549560547,
        157.630859375,
        0.8997195959091187,
        56.27149200439453,
        157.630859375,
        0.8680803775787354,
        37.36522674560547,
        209.623046875,
        0.80152428150177,
        51.54492950439453,
        207.259765625,
        0.8265056610107422
      ]))