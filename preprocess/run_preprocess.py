import sys
sys.path.append('.')

import glob
from preProcessor import HeatmapProcessor, PointCloudGenerator, AnnotationProcessor


if __name__ == '__main__': 
    processor = AnnotationProcessor(
    # processor = HeatmapProcessor(
        source_dir = "/root/raw_data/demo03/", 
        target_dir = "/root/proc_data/HuPR_demo/"
    )
    processor.run_processing()