import sys
sys.path.append('.')

import glob
from preProcessor import HeatmapProcessor, PointCloudGenerator, AnnotationProcessor


if __name__ == '__main__': 
    # processor = AnnotationProcessor(
    processor = HeatmapProcessor(
        source_dir = "/root/raw_data/demo/", 
        # target_dir = "/root/proc_data/HuPR_collected/", 
        target_dir = "/root/proc_data/HuPR_collected_full/", 
        full_view=True
    )
    processor.run_processing()