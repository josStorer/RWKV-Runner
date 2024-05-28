
base_model='/home/rwkv/JL/model/rwkv-x060-7b-world-v2.1-36%trained-20240413-ctx4k.pth'
lora_init='/home/rwkv/JL/out_model/nf4/init_lora.pth'
lora_checkpoint='/home/rwkv/JL/out_model/nf4/rwkv-0.pth'
output='/home/rwkv/JL/model/nf4-world.pth'
QUANT='nf4' #follow train
TYPE='lora'
Lora_alpha=128

python merge/merge.py --base_model $base_model \
--lora_init $lora_init \
--lora_checkpoint $lora_checkpoint \
--output $output \
--quant $QUANT \
--type $TYPE \
--lora_alpha $Lora_alpha