import cupy as cp
import numpy as np

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
