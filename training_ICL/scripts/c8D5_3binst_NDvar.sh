mask=True
attr="ICL-Llama3.2-3bInst-c8D5"

train_ds="./data/synth_data/MLT/ICLPT_c8_D5_nD1000000_N1_S42_LCont10-20/HO0.0"
test_ds="./data/synth_data/MLT/ICLPT_c8_D5_nD1000000_N1_S42_LCont10-20/HO0.0"
model_name_or_path="meta-llama/Llama-3.2-3B-Instruct"
BATCH_SIZE=2

cqa_ratio=1.0
cqa_disjoint="True"
weight_decay=0.1

save_freq=2048
eval_freq=128

n_textbook=1000000
n_sample=$n_textbook

additional_args=""
dict_irrelevant_rate_curriculum="on-0.25"
think_token_curriculum="0.0@0.1-1.0@0.6"
think_token_direction="forward"
think_token_direction_per_level="forward"
CoT_on=True

bash scripts/train_ICLPT.sh \
    $attr \
    $train_ds \
    $test_ds \
    $save_freq \
    $eval_freq \
    $n_textbook \
    $n_sample \
    $model_name_or_path \
    $BATCH_SIZE \
    $CoT_on \
    $dict_irrelevant_rate_curriculum \
    $think_token_curriculum \
    $think_token_direction \
    $think_token_direction_per_level \
    $additional_args