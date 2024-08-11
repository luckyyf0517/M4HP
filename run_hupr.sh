#!/bin/bash

if [ "$#" -ge 1 ]; then
    VERSION_OR_CONFIG=$1
else
    echo "Usage: $1 <version_or_config>, $2 <if_eval>"
    exit 1
fi

EVAL_FLAG=""

if [ "$2" == "eval" ]; then
    EVAL_FLAG="--eval"
fi

echo python baselines/run_hupr.py --version $VERSION_OR_CONFIG --config $VERSION_OR_CONFIG.yaml --gpuIDs '[0,1,2,3]' $EVAL_FLAG

export OMP_NUM_THREADS=10 
python baselines/run_hupr.py --version $VERSION_OR_CONFIG --config $VERSION_OR_CONFIG.yaml --gpuIDs '[0,1,2,3]' $EVAL_FLAG --visDir /root/viz