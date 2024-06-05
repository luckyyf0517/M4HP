export OMP_NUM_THREADS=10 
python baselines/run_hupr_classification.py --version $1 --sampling_ratio 1 --config $1.yaml --gpuIDs '[0]' #--visDir /root/viz --eval 