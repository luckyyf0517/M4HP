export OMP_NUM_THREADS=15
export NUMEXPR_NUM_THREADS=5
python baselines/run_hupr.py --version train_hupr_07\
                            --visDir /root/viz\
                            --sampling_ratio 1\
                            --config mscsa_prgcn_demo.yaml\
                            --gpuIDs '[0,1]'