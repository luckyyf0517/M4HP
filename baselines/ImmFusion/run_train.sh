# echo 'model: AdaptiveFusion'
# python ./run_immfusion_lightning.py \
#     --output_dir output/immfusion \
#     --dataset mmBodyDataset \
#     --data_path datasets/mmBody \
#     --model AdaptiveFusion \
#     --input_feat_dim "2051, 512, 128" \
#     --hidden_feat_dim "1024, 256, 64" \
#     --output_feat_dim "3" \
#     --per_gpu_train_batch_size 10 \
#     --gpu_idx '[0, 1]' \
#     --train

# echo 'model: VarAutoEncoder'
# python ./run_immfusion_lightning.py \
#     --output_dir output/autoencoder-32-1024 \
#     --data_path datasets/mmBody \
#     --per_gpu_train_batch_size 10 \
#     --gpu_idx '[0, 1]' \
#     --learning_rate 1e-4 \
#     --pretrain_vae \
#     --vae_latent_dim 32 \
#     --vae_hidden_feat_dim 1024 \
#     --train
    
echo 'model: DiffusionFusion'
python ./run_immfusion_lightning.py \
    --output_dir output/fusediffusion-v6 \
    --data_path datasets/mmBody \
    --model DiffusionFusion \
    --input_feat_dim "2051, 512, 128" \
    --hidden_feat_dim "1024, 256, 64" \
    --output_feat_dim "3" \
    --per_gpu_train_batch_size 10 \
    --gpu_idx '[0,1,2,3]' \
    --learning_rate 1e-4 \
    --train \
    --eval_test_dataset \
    --fix_modalities \
    --wo_MMM \
