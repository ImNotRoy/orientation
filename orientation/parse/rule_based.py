import numpy as np
import math

from tqdm import tqdm
from orientation.utils.keypoint import vector_distance
from orientation.utils.keypoint import select_main_character_based_size, select_main_character_based_distance


openpose_kp_dir = {"nose": 0, "neck": 1, "right_shoulder": 2, "right_elbow": 3,
                   "right_wrist": 4, "left_shoulder": 5, "left_elbow": 6, "left_wrist": 7,
                   "right_hip": 8, "right_knee": 9, "right_ankle": 10, "left_hip": 11,
                   "left_knee": 12, "left_ankle": 13, "right_eye": 14, "left_eye": 15,
                   "right_ear": 16, "left_ear": 17}

openpose_line_pair = [(0, 1), (0, 14), (0, 15), (1, 2), (1, 5), (1, 8), (1, 11), (2, 3), (3, 4), (5, 6), (6, 7),
                      (8, 9), (9, 10), (11, 12), (12, 13), (14, 16), (15, 17)]

head_ky_dir = {"nose": 0, "right_eye": 14, "left_eye": 15, "right_ear": 16, "left_ear": 17}


class Keypoint2Angle:

    def __init__(self):
        pass

    def vector_included_angle(self, v1, v2):
        '''
        detail:
            calculate vector included angle of v1 and v2
        args:
            v1, v2 = [x1,x2], [y1, y2]
        return:
            theta
        '''
        if v1[0] == 0 and v1[1] > 0:
            theta = 0.0
        elif v1[0] == 0 and v1[1] < 0:
            theta = 180.0
        else:
            # 弧度制theta
            cosine = np.clip(np.dot(v1, v2)
                              / (np.linalg.norm(v1, ord=2) * np.linalg.norm(v2, ord=2)), -1, 1)
            theta = math.acos(cosine)
            # 角度制theta
            theta = theta * 180 / np.pi
            if v1[0] > 0:
                theta
            else:
                theta = 360.0 - theta
        return theta

    def get_keypoint_position(self, pose, keypoint_name):
        '''
        detail:
            get keypoint position and confidence from given pose array
        args:
            pose -- float[], a array of pose info
            keypoint_name -- str, a name of keypoint which is a key in openpose_kp_dir 
        return:
            pnt -- (float, float), point position 
            confi -- float, confidence of this keypoint 
        '''

        pnt = (pose[3 * (openpose_kp_dir[keypoint_name])],
               pose[3 * (openpose_kp_dir[keypoint_name]) + 1])
        confi = pose[3 * (openpose_kp_dir[keypoint_name]) + 2]

        return pnt, confi

    def get_np_vector_from_pnt(self, p1, p2):
        '''
        detail:
            return a numpy array format vector, point from point2 to point1
        input:
            v1, v2 -- (float, float)
        return:
            numpy array
        '''
        return np.asarray([p2[0] - p1[0], p2[1] - p1[1]])

    def shoulder_hip_calculator(self, pose, img_shape):
        '''
        detail:
            Calculating the angle between the vector which conect left and right shoulder,
            and the vector which in the middle of two vetors, first connect neck and left hip, 
            second connect neck and right hip.
        input:
            pose -- a float array
            img_shape: width and height of image
        return:
            theta -- a float varible
            confi -- a float varible 
        '''
        l_shu_pnt, l_shu_confi = self.get_keypoint_position(pose, "left_shoulder")
        r_shu_pnt, r_shu_confi = self.get_keypoint_position(pose, "right_shoulder")
        l_hip_pnt, l_hip_confi = self.get_keypoint_position(pose, "left_hip")
        r_hip_pnt, r_hip_confi = self.get_keypoint_position(pose, "right_hip")
        neck_pnt, neck_confi = self.get_keypoint_position(pose, "neck")

        confi = min(l_shu_confi, r_shu_confi, l_hip_confi, r_hip_confi, neck_confi)

        v_i = self.get_np_vector_from_pnt(l_shu_pnt, r_shu_pnt)
        v_left_down = self.get_np_vector_from_pnt(l_hip_pnt, neck_pnt)
        v_right_down = self.get_np_vector_from_pnt(r_hip_pnt, neck_pnt)
        v_down = v_left_down + v_right_down
        # if the shoulder key-point exist, we can classify the people based on upper body. add by Yang shan.2021.11.12
        if (min(l_shu_confi, r_shu_confi) != 0) & (min(l_hip_confi, r_hip_confi) == 0):
            pos1 = (img_shape['width'] / 2, img_shape['height'])
            pos2 = (img_shape['width'] / 2, 0)
            v_down = self.get_np_vector_from_pnt(pos1, pos2)
            confi = min(l_shu_confi, r_shu_confi)

        return self.vector_included_angle(v_i, v_down), confi

    def side_view(self, pose, img_shape):  # Add by yang
        '''
        Differentiate the side view based proportion that shoulder length divide length from neck to hip with non-zero
        confidence and direction of nose vector
        :param pose:
        :return:
        '''
        left_sho_pnt, _ = self.get_keypoint_position(pose, 'left_shoulder')
        right_sho_pnt, _ = self.get_keypoint_position(pose, 'right_shoulder')
        l_hip_pnt, l_hip_confi = self.get_keypoint_position(pose, "left_hip")
        r_hip_pnt, r_hip_confi = self.get_keypoint_position(pose, "right_hip")
        neck_pnt, _ = self.get_keypoint_position(pose, "neck")
        left_sho_pnt = np.array(left_sho_pnt)
        right_sho_pnt = np.array(right_sho_pnt)
        l_hip_pnt = np.array(l_hip_pnt)
        r_hip_pnt = np.array(r_hip_pnt)
        neck_pnt = np.array(neck_pnt)

        shoulder_len = vector_distance(left_sho_pnt, right_sho_pnt)
        # Compute direction of side view
        head_ky_name = ["nose", "right_eye", "left_eye", "right_ear", "left_ear"]
        nose_angle = None
        for name in head_ky_name:
            pnt, confi = self.get_keypoint_position(pose, name)  # TODO Which one of the left and right eyes or ear.
            if pnt[0] != 0:
                head_pnt = pnt
                head_vec = self.get_np_vector_from_pnt(neck_pnt, head_pnt)
                shoulder_vec = self.get_np_vector_from_pnt(neck_pnt, left_sho_pnt)
                nose_angle = self.vector_included_angle(head_vec, shoulder_vec)
                break
        if nose_angle == None:
            nose_angle = float('inf')  # unknown angle
        # Compute proportion between shoulder length and neck hip length
        if (l_hip_confi == 0) & (r_hip_confi != 0):
            neck_hip_len = vector_distance(r_hip_pnt, neck_pnt)
            return shoulder_len / neck_hip_len, nose_angle
        elif (l_hip_confi != 0) & (r_hip_confi == 0):
            neck_hip_len = vector_distance(l_hip_pnt, neck_pnt)
            return shoulder_len / neck_hip_len, nose_angle
        elif (l_hip_confi != 0) & (r_hip_confi != 0):
            neck_hip_len = vector_distance(l_hip_pnt, neck_pnt)
            return shoulder_len / neck_hip_len, nose_angle
        elif (l_hip_confi == 0) & (r_hip_confi == 0):
            return shoulder_len / img_shape['width'], nose_angle

    def run(self, pose_data, img_shape, confi_threshold=0.0, angle_threshold=60, side_view_confi_thre=0.4):
        # from args get arguments
        view = None
        theta, confi = self.shoulder_hip_calculator(pose_data, img_shape)
        ppt, nose_angle = self.side_view(pose_data, img_shape)
        # 根据置信度分类,如果置信度如果低于阈值，则归位未知视角类别
        if confi <= confi_threshold:
            view = "unknown_view"

            return view
        # 根据肩膀的长度和颈部到髋关节的长度的比例选择侧面，然后再根据角度区分背面和正面
        # side view
        if ppt < side_view_confi_thre:
            if 0 < nose_angle < 180:
                view = "right_view"
                return view
            elif 360 > nose_angle >= 180:
                view = "left_view"
                return view
        # front_view
        if theta >= 270 - angle_threshold and theta <= 270 + angle_threshold:
            view = "front_view"
            return view
        # back_view
        elif theta >= 90 - angle_threshold and theta <= 90 + angle_threshold:
            view = "back_view"
            return view
        if view == None:
            view = "unknown_view"
            return view

    def get_orientation(self, keypoints_data,img_shape, confi_threshold=0.0, angle_threshold=60,
                 side_view_confi_thre=0.4):
        pose_data_list = []
        if len(keypoints_data['people']) == 0:  # return "unknown_view" when the pose_data is NONE
            orientation = "unknown_view"
            return orientation
        else:
            for i_people in keypoints_data['people']:
                pose_keypoints_2d = i_people['pose_keypoints_2d']
                pose_data_list.append(pose_keypoints_2d)
            w, h = img_shape['width'], img_shape['height']
            img_info = {'pose': pose_data_list, 'image_size': {'width': w, 'height': h}}
            # select main character in the image base on people size or distance between people and center of the image
            try:
                pose_data = select_main_character_based_size(
                    pose_data_list)
            except:
                # use the "distance" function when upper or lower body key-point("size" function) is not work.
                pose_data = select_main_character_based_distance(
                    img_info)
            orientation = self.run(pose_data, confi_threshold, angle_threshold,   # why k2g.run? 
                                  side_view_confi_thre,
                                  img_shape={'width': w, 'height': h})
            return orientation


k2a = Keypoint2Angle()


def get_orientation(image: np.ndarray, info: dict):
    keypoints = info["pose"]["keypoints"]
    height, width, _ = image.shape
    r_keys = ["front_view", "", "", "left_view", "right_view", "back_view", "unknown_view"]
    r = k2a.run(
        keypoints,
        {"height": height, "width": width},
    )
    if r not in r_keys:
        return len(r_keys) - 1
    return r_keys.index(r)


def test_dataset(dataset):
    return [
        get_orientation(image, info) for image, info in tqdm(dataset)
    ]
