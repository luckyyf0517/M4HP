export OMP_NUM_THREADS=10 
python baselines/run_hupr.py --version $1 --sampling_ratio 1 --config $2.yaml --gpuIDs '[2, 3]' 