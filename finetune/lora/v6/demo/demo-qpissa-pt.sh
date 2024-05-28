load_model='/home/rwkv/JL/model/rwkv-x060-7b-world-v2.1-36%trained-20240413-ctx4k.pth'
proj_dir='/home/rwkv/JL/out_model/nf4'
data_file='/home/rwkv/JL/data/roleplay'

QUANT='nf4'  #4bit nf4 fp4 none
svd_niter=4  
lora_r=64

n_layer=32
n_embd=4096

micro_bsz=4
epoch_save=1
epoch_steps=1000
ctx_len=1024


python train.py --proj_dir $proj_dir --data_file $data_file \
--data_type binidx --vocab_size 65536 \
--ctx_len $ctx_len --epoch_steps $epoch_steps --epoch_count 20 --epoch_begin 0 --epoch_save $epoch_save --micro_bsz $micro_bsz \
--n_layer $n_layer --n_embd $n_embd \
--pre_ffn 0 --head_qk 0 --lr_init 5e-5 --lr_final 5e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
--accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_1 --grad_cp 1 \
--my_testing "x060" \
--lora_load rwkv-0 --lora --lora_r $lora_r --lora_alpha 128 --lora_dropout 0.01 --lora_parts=att,ffn,time,ln \
--PISSA --svd_niter $svd_niter \
--quant $QUANT