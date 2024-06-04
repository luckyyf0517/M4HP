import sys
sys.path.append('.')

import glob
from preProcessor import AnnotationProcessor, HeatmapProcessor, SegmentAnnotationProcessor


if __name__ == '__main__': 
    # processor = SegmentAnnotationProcessor(
    processor = AnnotationProcessor(
        source_dir = "/root/raw_data/demo", 
        target_dir = "/root/proc_data/HuPR_collected/", 
    )
    processor.run_processing()