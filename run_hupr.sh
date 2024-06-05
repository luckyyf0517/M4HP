export OMP_NUM_THREADS=10 
python baselines/run_hupr.py --version $1 --sampling_ratio 1 --config $1.yaml --gpuIDs '[0,1,2,3]' --eval #--visDir /root/viz