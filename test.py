import ctypes

lib_path = "/root/anaconda3/envs/mmwave/lib/python3.8/site-packages/onnxruntime/capi/libonnxruntime_providers_cuda.so"
ctypes.cdll.LoadLibrary(lib_path)