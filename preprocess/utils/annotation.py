import json
import numpy as np
from tqdm import tqdm

def read_skeleton(skeleton2d_file, type='2d'):
    annotation_src = json.load(open(skeleton2d_file, 'r'))
    annotation_tgt = []
    for idx_frame in tqdm(annotation_src): 
        src_frame = annotation_src[idx_frame]
        # handle empty frames
        if 'skeleton' not in src_frame:
            try_frame = int(idx_frame) + 1
            while int(try_frame) < len(annotation_src): 
                src_frame = annotation_src['%04d' % try_frame]
                if 'skeleton' in src_frame: 
                    break
                try_frame += 1
            assert try_frame - int(idx_frame) < 30, 'No skeleton found for frame %s' % idx_frame
        joints = []
        if type == '2d':
            for i, point in enumerate(src_frame['skeleton']['joints2D']): 
                joints.append([point['position']['x'], point['position']['y']])
        elif type == '3d': 
            for i, point in enumerate(src_frame['skeleton']['joints']): 
                joints.append([point['position']['x'], point['position']['y'], point['position']['z']])
        joints = np.array(joints)
        if type == '2d':
            joints = scale_joints_to_2d(joints)           
        joints = mapping_joints_to_annotation(joints)
        bbox = [joints[:, 0].min(), joints[:, 1].min(), joints[:, 0].max(), joints[:, 1].max()]
        annotation_frame = {'frame': '%s' % idx_frame, 'joints': joints.tolist(), 'bbox': bbox}
        annotation_tgt.append(annotation_frame)
    return annotation_tgt

def scale_joints_to_2d(joints):
    width, height = (640, 576)
    left = (width - 576) / 2
    top = (height - 576) / 2
    scale_x = 256 / 576
    scale_y = 256 / 576
    new_joints = np.zeros((len(joints), 2))
    for i, joint in enumerate(joints):
        new_joints[i, 0] = (joint[0] - left) * scale_x
        new_joints[i, 1] = (joint[1] - top) * scale_y
    new_joints = np.clip(new_joints, 0, 255)
    return new_joints

def mapping_joints_to_annotation(joints): 
    map = [22, 23, 24, 18, 19, 20, 3, 27, 5, 6, 7, 12, 13, 14]
    new_joints = np.zeros((14, joints.shape[1]))
    for i in range(14):
        new_joints[i] = joints[map[i]]
    return new_joints
