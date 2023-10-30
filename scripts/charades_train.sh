config=pool_charades_16x16_k5l8
version=d3g

config_file=../configs/$config\.yaml
output_dir=../outputs/$config/$version

gpus=0,1,2,3
gpun=4
master_addr=127.0.0.9
master_port=29519

CUDA_VISIBLE_DEVICES=$gpus python3 -m torch.distributed.launch \
--nproc_per_node=$gpun --master_addr $master_addr --master_port $master_port \
../train_net.py \
--config-file $config_file \
OUTPUT_DIR $output_dir 

