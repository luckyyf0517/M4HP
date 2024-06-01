import sys
sys.path.append('.')

import glob
from preProcessor import HeatmapProcessor, PointCloudGenerator, AnnotationProcessor


if __name__ == '__main__': 
    processor = HeatmapProcessor(
        source_dir = "/root/raw_data/demo/", 
        target_dir = "/root/proc_data/HuPR_collected/"
    )
    processor.run_processing()