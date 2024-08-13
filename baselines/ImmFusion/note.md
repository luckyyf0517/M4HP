run eval code: 

python ./run_immfusion.py --output_dir output/immfusion --resume_checkpoint output/immfusion/checkpoint --dataset mmBodyDataset --data_path datasets/mmBody --mesh_type smplx --model AdaptiveFusion --test_scene lab1

run train code: 

python ./run_immfusion.py --output_dir output/immfusion --dataset mmBodyDataset --data_path datasets/mmBody --mesh_type smplx --model AdaptiveFusion --per_gpu_train_batch_size 10 --train