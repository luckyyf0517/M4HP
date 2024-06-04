export OMP_NUM_THREADS=10; \
export NUMEXPR_NUM_THREADS=3; 
python baselines/run_hupr.py --version $1 --sampling_ratio 1 --config $2.yaml --gpuIDs '[0, 1, 2, 3]'