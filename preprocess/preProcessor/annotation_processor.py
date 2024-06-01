import sys
sys.path.append('.')

import os
import json
from tqdm import tqdm

from .preprocessor import PreProcessor
from .utils import fromKinect2HuPR

class AnnotationProcessor(PreProcessor):
    def __init__(self, source_dir, target_dir):
        super().__init__(source_dir, target_dir)
    
    def run_processing(self): 
        for seq_name in self.source_seqs:
            print('Processing', seq_name)
            with open(os.path.join(self.source_dir, seq_name, 'kinect', 'skeleton2d.json'), 'r') as f: 
                annotation_src = json.load(f)
            annotation_tgt = []
            for idx_frame in tqdm(annotation_src): 
                src_frame = annotation_src[idx_frame]
                xs = []
                ys = []
                for point in src_frame['skeleton']['joints2D']: 
                    xs.append(point['position']['x'])
                    ys.append(point['position']['y'])
                new_xs, new_ys = fromKinect2HuPR(xs=xs, ys=ys)           
                annotation_frame = {'image': '%s.jpg' % idx_frame, 'joints': []}
                for x, y in zip(new_xs, new_ys): 
                    annotation_frame['joints'].append([x, y]) 
                annotation_tgt.append(annotation_frame)
            
            if not os.path.exists(os.path.join(self.target_dir, seq_name)):
                os.makedirs(os.path.join(self.target_dir, seq_name))
            with open(os.path.join(self.target_dir, seq_name, 'annotation.json'), 'w') as f: 
                json.dump(annotation_tgt, f)