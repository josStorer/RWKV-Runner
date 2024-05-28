base_model='/home/rwkv/JL/model/RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth'
state_checkpoint='/home/rwkv/JL/out_model/state/rwkv-9.pth'
output='/home/rwkv/JL/model/state-0.pth'


python merge/merge_state.py --base_model $base_model \
--state_checkpoint $state_checkpoint \
--output $output