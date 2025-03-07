mask=True
attr="CEL-C10D5-fakedesc"

train_ds_IM="./data/synth_data/MLT/CEL_c10_D5_nD1_N1000000_S0_LCont10-20/HO0.0"
test_ds_IM="./data/synth_data/MLT/CEL_c10_D5_nD1_N1000000_S0_LCont10-20/HO0.0"
train_ds_ICL="./data/synth_data/MLT/ICLPT_c10_D5_nD1000000_N1_S42_LCont10-20/HO0.0"
test_ds_ICL="./data/synth_data/MLT/ICLPT_c10_D5_nD1000000_N1_S42_LCont10-20/HO0.0"
model_name_or_path= #TODO: put the ICLPT model here
BATCH_SIZE=2

cqa_ratio=1.0
cqa_disjoint="True"
weight_decay=0.1

save_freq=8192
eval_freq=128

for icl_im_datamix in  "False" #"True"
do

for weight_decay in 0
do

n_textbook_IM=1

for n_sample in 1000000 500000 250000 100000 10000
do

n_sample_IM=$n_sample
n_sample_ICL=$n_sample
n_textbook_ICL=$n_sample

for additional_args in "--fake_dictionary True"
do

for dict_irrelevant_rate_curriculum in "on-0.25"
do

for dictionary_mask_rate_curriculum in "0.0@0.0-1.0@0.6"
do

for dictionary_mask_token_direction in  "even_disjoint_random"
do

for dictionary_mask_config in "1-1-1-1-1"
do


CoT_on=True
job_name=$attr"_"$n_sample"_DR"$icl_im_datamix"_"$dict_irrelevant_rate_curriculum"_"$dictionary_mask_rate_curriculum"_"$dictionary_mask_token_direction"_"$dictionary_mask_config"_"$CoT_on"_"$additional_args

echo $job_name

bash scripts/train_CEL.sh \
    $attr \
    $train_ds_IM \
    $test_ds_IM \
    $save_freq \
    $eval_freq \
    $n_textbook_IM \
    $n_sample_IM \
    $model_name_or_path \
    $BATCH_SIZE \
    $CoT_on \
    $dict_irrelevant_rate_curriculum \
    $dictionary_mask_rate_curriculum \
    $dictionary_mask_token_direction \
    $dictionary_mask_config \
    $train_ds_ICL \
    $test_ds_ICL \
    $n_textbook_ICL \
    $n_sample_ICL \
    $icl_im_datamix \
    $weight_decay \
    $additional_args

done
done
done
done
done
done
done
done
