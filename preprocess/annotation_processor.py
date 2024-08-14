import sys
sys.path.append('.')

import os
import json
import glob
from tqdm import tqdm

from preprocess.utils.annotation import read_skeleton

class AnnotationProcessor():
    def __init__(self):
        self.source_paths = sorted(glob.glob('M4HPDataset/Annot_files/*/*/*/kinect/'))
        print('Found', len(self.source_paths), 'source paths')
    
    def run_processing(self): 
        for source_path in self.source_paths:
            print('Processing', source_path)
            
            skeleton2d_file = os.path.join(source_path, 'skeleton2d.json')
            skeleton3d_file = os.path.join(source_path, 'skeleton3d.json')
            joints2d = read_skeleton(skeleton2d_file, '2d')
            joints3d = read_skeleton(skeleton3d_file, '3d')
            
            target_path = source_path.replace('kinect', 'annotation')
            os.makedirs(target_path, exist_ok=True)
            json.dump(joints2d, open(os.path.join(target_path, 'skeleton2d.json'), 'w'))
            json.dump(joints3d, open(os.path.join(target_path, 'skeleton3d.json'), 'w'))
            
            
if __name__ == '__main__':
    processor = AnnotationProcessor()
    processor.run_processing()