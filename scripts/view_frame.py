import sys
sys.path.append('.')

import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from preprocess.preProcessor.utils import plot_heatmap2D


if __name__ == '__main__': 
    seq_name = 'single_2'
    
    data_dir = './data/HuPR_official/%s/hori' % seq_name 
    data_list = sorted(os.listdir(data_dir))

    for data_name in tqdm(data_list): 
        data_hori = np.load(os.path.join(data_dir, data_name))   
        data_vert = np.load(os.path.join(data_dir.replace('hori', 'vert'), data_name))   

        plt.clf()
        plt.subplot(121)
        plot_heatmap2D(data_hori, axes=(1, 2), title='Range-Angle View')
        plt.subplot(122)
        plot_heatmap2D(data_vert, axes=(1, 2), title='Range-Elevation View', transpose=True)
        
        os.makedirs('./viz/%s/heatmap' % seq_name, exist_ok=True)
        plt.savefig('./viz/%s/heatmap/%s.png' % (seq_name, data_name[:9]))