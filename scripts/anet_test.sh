config=pool_activitynet_64x64_k9l4
version=d3g

config_file=../configs/$config\.yaml
output_dir=../outputs/inferences/$config/$version
weight_file=../outputs/$config/$version/best_model.pth

batch_size=48
gpus=0,1,2,3
gpun=4
master_addr=127.0.0.2
master_port=29578

CUDA_VISIBLE_DEVICES=$gpus python3 -m torch.distributed.launch \
--nproc_per_node=$gpun --master_addr $master_addr --master_port $master_port \
../test_net.py --config-file $config_file --ckpt $weight_file OUTPUT_DIR $output_dir TEST.BATCH_SIZE $batch_size