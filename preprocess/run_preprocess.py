import sys
sys.path.append('.')

import glob
from preProcessor import AnnotationProcessor, HeatmapProcessor


if __name__ == '__main__': 
    processor = AnnotationProcessor(
        source_dir = "/root/raw_data/demo", 
        target_dir = "/root/proc_data/HuPR_collected/", 
        # target_dir = "/root/proc_data/HuPR_collected_full/", 
        # full_view=True
    )
    processor.run_processing()