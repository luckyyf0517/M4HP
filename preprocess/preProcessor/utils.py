import os
import mpld3
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objs as go
import plotly.io as pio


def to_numpy(data): 
    if isinstance(data, cp.ndarray): 
        return data.get()
    else: 
        return data
    
    
def find_peak(data_org, peak_source=None):
    if peak_source is None: 
        peak_source = data_org
    data_max = np.argmax(np.abs(data_org), axis=0)
    data_peak = np.zeros_like(data_max, dtype=np.complex_) 
    for i in range(len(data_max)):
        data_peak[i] = peak_source[data_max[i], i]
    return data_max, data_peak
    

def plot_pointcloud(pointcloud, target_path_folder=None):
    pointcloud = to_numpy(pointcloud)
    print(pointcloud.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2])     
    mpld3.show()
    

def plot_heatmap2D(data_array, axes=None, title=None, transpose=False): 
    assert len(axes) == 2, 'Num of axes must be 2'
    data_array = to_numpy(data_array)
    axis_sum = [dim for dim in range(data_array.ndim) if dim not in axes]
    data_map = np.abs(data_array)
    data_map = np.sum(data_map, axis=tuple(axis_sum))
    plt.imshow(data_map if not transpose else data_map.T)
    if title is not None: 
        plt.title(title)
    
def fromKinect2HuPR(xs, ys):
    width, height = (640, 576)
    left = (width - 576) / 2
    top = (height - 576) / 2
    scale_x = 256 / 576
    scale_y = 256 / 576
    new_xs = []
    new_ys = []
    for x, y in zip(xs, ys):
        new_x = (x - left) * scale_x
        new_y = (y - top) * scale_y
        new_x = max(0, min(new_x, 255))
        new_y = max(0, min(new_y, 255))
        new_xs.append(new_x)
        new_ys.append(new_y)
    map = [22, 23, 24, 18, 19, 20, 3, 27, 5, 6, 7, 12, 13, 14]
    hupr_xs = []
    hupr_ys = []
    for i in map:
        hupr_xs.append(new_xs[i])
        hupr_ys.append(new_ys[i])
    return hupr_xs, hupr_ys
