import sys
sys.path.append('.')

from preProcessor import HeatmapProcessor, PointCloudGenerator


if __name__ == '__main__': 
    processor = PointCloudGenerator(
        source_dir = r"/root/raw_data/demo/", 
        source_seqs = [
            # '2024-05-29-22-22-05-443181',   # left
            # '2024-05-29-22-21-29-074018',   # right
            # '2024-05-29-22-22-37-027792',    # T
            # '2024-05-29-23-38-57-931262',   # L
            # '2024-05-29-23-40-00-290270',   # R
            # '2024-05-29-23-41-25-579382',   # L far
            # '2024-05-29-23-42-19-849302',   # R far
            # '2024-05-29-23-42-58-051479',   # T
            # '2024-05-30-15-16-03-529798', 
            # '2024-05-30-15-15-28-511850', 
            # '2024-05-30-17-09-22-176764', 
            # '2024-05-30-17-08-49-325422', 
            '2024-05-30-17-58-57-870565', 
            # '2024-05-30-17-59-43-324699', 
        ], 
        target_dir = r"/root/proc_data/HuPR/"
    )
    processor.run_processing()