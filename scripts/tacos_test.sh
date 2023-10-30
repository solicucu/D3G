config=pool_tacos_128x128_k5l8
version=d3g

config_file=../configs/$config\.yaml
output_dir=../outputs/inferences/$config/$version
weight_file=../outputs/$config/$version/best_model.pth

batch_size=16
gpus=0,1
gpun=2
master_addr=127.0.0.8
master_port=29578

CUDA_VISIBLE_DEVICES=$gpus python3 -m torch.distributed.launch \
--nproc_per_node=$gpun --master_addr $master_addr --master_port $master_port \
../test_net.py --config-file $config_file --ckpt $weight_file OUTPUT_DIR $output_dir TEST.BATCH_SIZE $batch_size