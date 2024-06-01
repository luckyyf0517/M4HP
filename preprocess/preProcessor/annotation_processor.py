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
            if 'deprecated' in seq_name: 
                continue
            print('Processing', seq_name)
            with open(os.path.join(self.source_dir, seq_name, 'kinect', 'skeleton2d.json'), 'r') as f: 
                annotation_src = json.load(f)
            annotation_tgt = []
            for idx_frame in tqdm(annotation_src): 
                src_frame = annotation_src[idx_frame]
                xs = []
                ys = []
                
                if 'skeleton' not in src_frame:
                    try_frame = int(idx_frame) + 1
                    while int(try_frame) < len(annotation_src): 
                        src_frame = annotation_src['%04d' % try_frame]
                        if 'skeleton' in src_frame: 
                            break
                        try_frame += 1
                    assert try_frame - int(idx_frame) < 10, 'No skeleton found for frame %s' % idx_frame
                
                for point in src_frame['skeleton']['joints2D']: 
                    xs.append(point['position']['x'])
                    ys.append(point['position']['y'])
                new_xs, new_ys = fromKinect2HuPR(xs=xs, ys=ys)           
                annotation_frame = {'image': '%s.jpg' % idx_frame, 'joints': [], 'bbox': [min(xs), min(ys), max(xs), max(ys)]}
                for x, y in zip(new_xs, new_ys): 
                    annotation_frame['joints'].append([x, y]) 
                annotation_tgt.append(annotation_frame)
            
            if not os.path.exists(os.path.join(self.target_dir, seq_name)):
                os.makedirs(os.path.join(self.target_dir, seq_name))
            with open(os.path.join(self.target_dir, seq_name, 'annotation.json'), 'w') as f: 
                json.dump(annotation_tgt, f)