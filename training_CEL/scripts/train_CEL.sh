DEV=0
n_gpus=1

PROJ_DIR="./"
RUN_DIR=$PROJ_DIR

data_seed=42
eval_n_textbook=1
eval_n_sample=256
NUM_EPOCHS=1

attr=$1
train_path_IM=$2
eval_path_IM=$3
save_freq=$4
eval_freq=$5
train_n_textbook_IM=$6
train_n_sample_IM=$7
MODEL=$8
BATCH_SIZE=$9
COT_ON=${10}

dict_irrelevant_rate_curriculum=${11}
dictionary_mask_rate_curriculum=${12}
dictionary_mask_token_direction=${13}
dictionary_mask_config=${14}

train_path_ICL=${15}
eval_path_ICL=${16}
train_n_textbook_ICL=${17}
train_n_sample_ICL=${18}

ICL_datamix=${19}
weight_decay=${20}

additional_args=${21}

cnt=0
seed=0
lr=1e-4

hid_size=$(( head_dim * n_heads ))
int_size=$(( hid_size * 5504 / 2048))

device_id=$(( $cnt % $n_gpus ))
cnt=$(( $cnt + 1 ))

NET_BATCH_SIZE=64
ACCU_STEPS=$(($NET_BATCH_SIZE / $BATCH_SIZE))
# vocab_size=128

subdir_log_name="ds"$train_n_textbook_IM"_"$train_n_sample_IM"_E"$NUM_EPOCHS"_WD"$weight_decay"_DR"$ICL_datamix"_DM"$dictionary_mask_rate_curriculum"_DMD"$dictionary_mask_token_direction"_DMC"$dictionary_mask_config"_dIP"$dict_irrelevant_rate_curriculum"_CoT"$COT_ON"_"$attr"_"$additional_args"_DR"$train_n_sample_ICL"_"$train_n_textbook_ICL

echo "subdir_log_name: "$subdir_log_name

OUTPUT_DIR=$RUN_DIR"/data/checkpoints/MLT_CEL/"$attr"/"$subdir_log_name
LOGGING_DIR=$RUN_DIR"/data/checkpoints/MLT_CEL/"$attr"/log/"$subdir_log_name
header="python $RUN_DIR/train_CEL.py"

echo ${header}

${header}  \
    --do_train True \
    --num_train_epochs $NUM_EPOCHS \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.06 \
    --weight_decay 0.0 \
    --weight_decay_pretrained $weight_decay \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --max_sequence_length 8192 \
    --logging_steps 16 \
    --save_strategy "steps" \
    --save_steps $save_freq \
    --eval_strategy "steps" \
    --overwrite_output_dir True \
    --optim adamw_torch \
    --gradient_checkpointing False \
    --seed $seed \
    --train_path_IM $train_path_IM \
    --eval_path_IM $eval_path_IM \
    --train_path_ICL $train_path_ICL \
    --eval_path_ICL $eval_path_ICL \
    --model_name_or_path $MODEL \
    --output_dir $OUTPUT_DIR \
    --logging_dir $LOGGING_DIR \
    --learning_rate $lr \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $ACCU_STEPS \
    --cache_dir $CACHE_DIR \
    --eval_steps $eval_freq \
    --per_device_eval_batch_size $((BATCH_SIZE * 2)) \
    --eval_accumulation_steps 16 \
    --preprocessing_num_workers 20 \
    --run_name $subdir_log_name \
    --project_name $attr \
    --train_n_textbook_IM $train_n_textbook_IM \
    --train_n_sample_IM $train_n_sample_IM \
    --eval_n_textbook_IM $eval_n_textbook \
    --eval_n_sample_IM $eval_n_sample \
    --train_n_textbook_ICL $train_n_textbook_ICL \
    --train_n_sample_ICL $train_n_sample_ICL \
    --eval_n_textbook_ICL $eval_n_sample \
    --eval_n_sample_ICL $eval_n_sample \
    --icl_im_datamix $ICL_datamix \
    --cot_on $COT_ON \
    --dict_irrelevant_rate_curriculum $dict_irrelevant_rate_curriculum \
    --think_token_curriculum "on" \
    --dictionary_mask_rate_curriculum $dictionary_mask_rate_curriculum \
    --dictionary_mask_token_direction $dictionary_mask_token_direction \
    --dictionary_mask_config $dictionary_mask_config \
    --bf16 True \
    --dictionary_mask_token_template "remove" \
    $additional_args;

    #if [ $DEV = 1 ]; then
    #    exit
    #fi

rm -rf $OUTPUT_DIR"/**/optimizer.pt"