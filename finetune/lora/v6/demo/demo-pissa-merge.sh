

base_model='/home/rwkv/JL/model/RWKV-x060-World-1B6-v2-20240208-ctx4096.pth'
lora_init='/home/rwkv/JL/out_model/nf4/init_lora.pth'
lora_checkpoint='/home/rwkv/JL/out_model/nf4/rwkv-0.pth'
output='/home/rwkv/JL/model/end-world.pth'
QUANT='nf4' #follow train
TYPE='pissa'

python merge/merge.py --base_model $base_model \
--lora_init $lora_init \
--lora_checkpoint $lora_checkpoint \
--output $output \
--quant $QUANT \
--type $TYPE 