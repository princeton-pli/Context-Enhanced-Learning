DEV=0
n_gpus=1

PROJ_DIR="./"
RUN_DIR=$PROJ_DIR

data_seed=42
CACHE_DIR=#TODO: Your Huggingface Cache DIR
eval_n_textbook=256
eval_n_sample=256
NUM_EPOCHS=1
weight_decay=0.1

attr=$1
train_path=$2
eval_path=$3
save_freq=$4
eval_freq=$5
train_n_textbook=$6
train_n_sample=$7 # 1000
MODEL=$8 # choose from "meta-llama/Llama-3.2-3B-Instruct" or "meta-llama/Meta-Llama-3.1-8B-Instruct"
BATCH_SIZE=$9
COT_ON=${10}

dict_irrelevant_rate_curriculum=${11}
think_token_curriculum=${12}
think_token_direction=${13}
think_token_direction_per_level=${14}

additional_args=${15}

# Please fill
TRAIN_FILE=$train_path
EVAL_FILE=$eval_path

cnt=0
seed=0
lr=1e-4

device_id=$(( $cnt % $n_gpus ))
cnt=$(( $cnt + 1 ))

NET_BATCH_SIZE=64
ACCU_STEPS=$(($NET_BATCH_SIZE / $BATCH_SIZE))

subdir_log_name="ds"$train_n_textbook"_"$train_n_sample"_E"$NUM_EPOCHS"_think"$think_token_curriculum"_"$think_token_direction"_"$think_token_direction_per_level"_dIP"$dict_irrelevant_rate_curriculum"_CoT"$COT_ON"_"$attr"_"$additional_args

echo "subdir_log_name: "$subdir_log_name

OUTPUT_DIR=$RUN_DIR"/data/checkpoints/MLT_ICL/"$attr"/"$subdir_log_name
LOGGING_DIR=$RUN_DIR"/data/checkpoints/MLT_ICL/"$attr"/log/"$subdir_log_name
header="python $RUN_DIR/train_ICLPT.py"

echo ${header}

${header}  \
    --do_train True \
    --num_train_epochs $NUM_EPOCHS \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.06 \
    --weight_decay $weight_decay \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --max_sequence_length 8192 \
    --logging_steps 16 \
    --save_strategy "steps" \
    --save_steps $save_freq \
    --eval_strategy "steps" \
    --eval_path $eval_path \
    --overwrite_output_dir True \
    --optim adamw_torch \
    --gradient_checkpointing False \
    --seed $seed \
    --train_path $train_path \
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
    --train_n_textbook $train_n_textbook \
    --train_n_sample $train_n_sample \
    --eval_n_textbook $eval_n_textbook \
    --eval_n_sample $eval_n_sample \
    --cot_on $COT_ON \
    --dict_irrelevant_rate_curriculum $dict_irrelevant_rate_curriculum \
    --think_token_curriculum $think_token_curriculum \
    --think_token_direction $think_token_direction \
    --think_token_direction_per_level $think_token_direction_per_level \
    --dictionary_mask_rate_curriculum "off" \
    --bf16 True \
    $additional_args;

    #if [ $DEV = 1 ]; then
    #    exit
    #fi

rm -rf $OUTPUT_DIR"/**/optimizer.pt"