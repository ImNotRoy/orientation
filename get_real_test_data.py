import json
import pickle

with open('/Users/longruihan/代码/orientation/data/test.json', 'r') as f:
    keypoints_dict = json.load(f)
test_view = pickle.load(open('/Users/longruihan/代码/orientation/data/test_res_rule.pickle', 'rb'))
new_dict = []
for info in keypoints_dict:
    keypoint_list = info["pose"]["keypoints"]
    if max(keypoint_list) == 0:
        continue

    if info['filename'].split('.')[0] not in test_view:
        continue

    if info["orientation"] == 1:
        info["orientation"] = 0

    if info["orientation"] == 2:
        info["orientation"] = 0
    new_dict.append(info)

res = json.dumps(new_dict)

with open('/Users/longruihan/代码/orientation/data/dml_test.json', 'w') as f:
    f.write(res)
