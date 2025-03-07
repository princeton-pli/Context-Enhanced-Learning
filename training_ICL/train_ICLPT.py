# train llama models

import torch
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers import HfArgumentParser, DataCollatorForSeq2Seq
from typing import Callable, Dict, Optional, Union, List
from components.all_arguments import ModelArguments, TrainingArguments, DataArguments, CurriculumArguments
from components.data_collator import DataCollator_masked_doc, DataCollator_unmasked
from components.data_utils import encode_dataset
from components.model_utils import add_new_tokens, archive_datasets
from components.prompts import TEMPLATE
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy

from datasets import  Dataset, concatenate_datasets
from dataclasses import asdict
from components.CE_trainer import CETrainer
import functools
import json
import os
import pandas as pd
import pickle
import numpy as np
import random
from tqdm import tqdm
import glob
from transformers import set_seed
from datasets import load_from_disk
import evaluate

from peft import get_peft_model, LoraConfig

import Levenshtein

from IPython import embed
import pdb

def preproc_logits_argmax(logits, labels):
    # For classification, we typically use argmax to get the predicted class
    predictions = logits.argmax(dim=-1)
    return predictions

parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, CurriculumArguments))

model_args, training_args, data_args, curriculum_args = parser.parse_args_into_dataclasses()    

print("start training")
#if training_args.project_name != "":
wandb_dir = training_args.output_dir
os.makedirs(wandb_dir, exist_ok=True)

os.environ["WANDB_PROJECT"]=training_args.project_name
os.environ["WANDB_DIR"]=wandb_dir
os.environ["WANDB_MODE"]="offline"

# get pid
pid = os.getpid()
training_args.pid = pid
# set seed
seed = training_args.seed
training_args.remove_unused_columns = False
set_seed(seed)

tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir, legacy=True)
tokenizer.pad_token = tokenizer.eos_token


train_dataset, raw_train_ds = encode_dataset(
    tokenizer, 
    'train',
    data_args=data_args,
    curriculum_args=curriculum_args
)

eval_datasets_indist_Ton, _ = encode_dataset(
    tokenizer, 
    'train_Ton',
    data_args=data_args,
    curriculum_args=curriculum_args,
    dict_ip='on',
    think_rate='on'
)

eval_datasets_indist_Toff, _ = encode_dataset(
    tokenizer, 
    'train_Ton',
    data_args=data_args,
    curriculum_args=curriculum_args,
    dict_ip='on',
    think_rate='off'
)

eval_datasets_outdist_Ton, _ = encode_dataset(
    tokenizer, 
    'test_Ton',
    data_args=data_args,
    curriculum_args=curriculum_args,
    dict_ip='on',
    think_rate='on'
)

eval_datasets_outdist_Toff, _ = encode_dataset(
    tokenizer, 
    'test_Toff',
    data_args=data_args,
    curriculum_args=curriculum_args,
    dict_ip='on',
    think_rate='off'
)

eval_datasets_outdist_Ton_noip, _ = encode_dataset(
    tokenizer, 
    'test_Ton_no_ip',
    data_args=data_args,
    curriculum_args=curriculum_args,
    dict_ip='off',
    think_rate='on'
)

eval_datasets_outdist_Toff_noip, _ = encode_dataset(
    tokenizer, 
    'test_Toff_no_ip',
    data_args=data_args,
    curriculum_args=curriculum_args,
    dict_ip='off',
    think_rate='off'
)




# embed()
n_languages = len(raw_train_ds[0][0]['dictionaries']) + 1
response_start = ' ' + TEMPLATE.cot_template.format(lang=n_languages, lang_str="")[:-1]
response_start_ids = tokenizer.encode(response_start, add_special_tokens=False)
print("Response start identifier: {} - {}".format(response_start, response_start_ids))
# meta_align_check_translate(train_meta, test_meta)

eval_datasets = {
    'indist_Ton': eval_datasets_indist_Ton,
    'indist_Toff': eval_datasets_indist_Toff,
    'outdist_Ton': eval_datasets_outdist_Ton,
    'outdist_Toff': eval_datasets_outdist_Toff,
    'outdist_Ton_noip': eval_datasets_outdist_Ton_noip,
    'outdist_Toff_noip': eval_datasets_outdist_Toff_noip
}

archive_datasets(train_dataset, eval_datasets, tokenizer, training_args.output_dir, n_samples=20)
for key in eval_datasets:
    print("Eval Dataset: {} - {}".format(key, len(eval_datasets[key])))

ref_config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir,)
model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir, torch_dtype=torch.bfloat16) # if training_args.bf16 else torch.float32)
trainer = CETrainer
# if 'llama' in model_args.model_name_or_path.lower():
tokenizer.pad_token_id = tokenizer.eos_token_id# = model_args.vocab_size - 1
# small_model_config = ref_config
# else:
# raise NotImplementedError

def prepare_eval_generation(inputs, labels, localized_answer_start_seq=None):
    instructions = []

    instruction_end = np.argmax(labels != -100, axis=1)
    answer_end = np.argmax(labels == tokenizer.eos_token_id, axis=1)

    answers = []
    answer_mask = np.zeros_like(inputs, dtype=int)

    localized_answers = []
    localized_answer_mask = np.zeros_like(inputs, dtype=int)

    for i, x in enumerate(inputs):
        instructions.append(x[:instruction_end[i]])
        answers.append(x[instruction_end[i]:answer_end[i] + 1])
        answer_mask[i][instruction_end[i]:answer_end[i] + 1] = 1
        assert x[answer_end[i]] == tokenizer.eos_token_id
    

    if localized_answer_start_seq is not None:
        # find the begining of localized_answer_start substring traversing from the end
        for i, x in enumerate(inputs):
            localized_answer_start = None
            for j in range(answer_end[i] - len(localized_answer_start_seq), instruction_end[i], -1):
                if (x[j:j + len(localized_answer_start_seq)] == localized_answer_start_seq).all():
                    localized_answer_start = j + len(localized_answer_start_seq)
                    break
            assert localized_answer_start is not None
            # -1 to remove the finish token (;)
            localized_answers.append(x[localized_answer_start:answer_end[i] - 1])
            localized_answer_mask[i][localized_answer_start:answer_end[i] - 1] = 1
    else:
        localized_answers = answers
        localized_answer_mask = answer_mask

    max_instruction_length = max([len(x) for x in instructions])
    instructions_aligned = np.zeros((len(answers), max_instruction_length))
    attention_masks = np.zeros_like(instructions_aligned)
    for i, x in enumerate(instructions):
        instructions_aligned[i, -len(x):] = x
        attention_masks[i, -len(x):] = 1

    return [
        torch.tensor(instructions_aligned).to(model.device).int(),
        torch.tensor(attention_masks).to(model.device).int(),
        answers,
        answer_mask,
        localized_answers,
        localized_answer_mask
    ]

def compute_metrics_translate(eval_preds):
    # Requires setting include_inputs_for_metrics=True
    predictions, labels, inputs = eval_preds
    metrics = {}

    # for generation evaluation
    instructions_aligned, attention_masks, answers, answer_mask, localized_answers, localized_answer_mask = prepare_eval_generation(inputs, labels, localized_answer_start_seq=response_start_ids)

    # for next token prediction metrics
    labels_shifted = labels[:, 1:]
    answer_mask_shifted = answer_mask[:, 1:]
    localized_answer_mask_shifted = localized_answer_mask[:, 1:]
    # predictions = np.argmax(logits[..., :-1, :], axis=-1)
    predictions = predictions[..., :-1]

    # == Autoregressive Evaluation
    match_dist = (predictions == labels_shifted) * answer_mask_shifted
    match_accuracy = (match_dist.sum(axis=1) / answer_mask_shifted.sum(axis=1)).mean()
    metrics['next_token_acc'] = match_accuracy

    # == Localized Autoregressive Evaluation
    localized_match_dist = (predictions == labels_shifted) * localized_answer_mask_shifted
    localized_match_accuracy = (localized_match_dist.sum(axis=1) / localized_answer_mask_shifted.sum(axis=1)).mean()
    metrics['next_token_acc_localized'] = localized_match_accuracy

    # Return if not doing generation evaluation
    if not training_args.eval_generation:
        return metrics

    # == Generation Evaluation
    print("Evaluating Generation")
    eval_batchsize = training_args.per_device_eval_batch_size
    max_ans_length = max([len(ans) for ans in answers])
    
    generated_results = []
    localized_generated_results = []

    for batch_count in range((len(instructions_aligned) - 1) // eval_batchsize + 1):
        outputs = model.generate(
            input_ids=instructions_aligned[batch_count * eval_batchsize: (batch_count + 1) * eval_batchsize],
            attention_mask=attention_masks[batch_count * eval_batchsize: (batch_count + 1) * eval_batchsize],
            max_new_tokens=int(max_ans_length * 1.5),
            do_sample=False,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        ).cpu().numpy()

        if batch_count == 0:
            output_strs = tokenizer.batch_decode(outputs)
            for s in output_strs[:2]:
                print(s, flush=True)

        # doing cutoff for general response
        responses = outputs[:, instructions_aligned.shape[-1]:]
        for response in responses:
            if tokenizer.eos_token_id not in response:
                generated_results.append(response)
            else:
                response_cutoff = response[:np.argmax(response == tokenizer.eos_token_id) + 1]
                generated_results.append(response_cutoff)

        # doing cutoff for localized response
        for response in responses:
            localized_ans_end = len(response)
            if tokenizer.eos_token_id in response:
                # doing -1 to remove the finish token (;)
                localized_ans_end = np.argmax(response == tokenizer.eos_token_id) - 1
            localized_ans_begin = 0
            for i in range(localized_ans_end - len(response_start_ids), 0, -1):
                if (response[i:i + len(response_start_ids)] == response_start_ids).all():
                    localized_ans_begin = i + len(response_start_ids)
            
            localized_response_cutoff = response[localized_ans_begin: localized_ans_end]
            localized_generated_results.append(localized_response_cutoff)
        
    levenshtein_distances = []
    for gen_ans, ans in zip(generated_results, answers):
        levenshtein_distances.append(Levenshtein.distance(gen_ans.tolist(), ans.tolist()) / len(ans))
    metrics['levenshtein_distances'] = np.mean(levenshtein_distances)
    
    localized_levenshtein_distances = []
    for gen_ans, ans in zip(localized_generated_results, localized_answers):
        localized_levenshtein_distances.append(Levenshtein.distance(gen_ans.tolist(), ans.tolist()) / len(ans))
    metrics['levenshtein_distances_localized'] = np.mean(localized_levenshtein_distances)

    return metrics

# embed()
### Define Trainer ###    
trainer_class = trainer
if training_args.doc_mask:
    collator = DataCollator_masked_doc
else:
    collator = DataCollator_unmasked
eval_collator = DataCollator_masked_doc

trainer = trainer_class(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_datasets,
    # tokenizer=tokenizer,
    data_collator=collator(tokenizer),
    eval_collator=eval_collator(tokenizer),
    compute_metrics=compute_metrics_translate,
    preprocess_logits_for_metrics=preproc_logits_argmax,  # Preprocess logits before passing to compute_metrics,
)

# trainer.save_tokenizer()
tokenizer.save_pretrained(training_args.output_dir)

# fsdp setup
if trainer.is_fsdp_enabled:
    # Override accelerate defaults
    trainer.accelerator.state.fsdp_plugin.limit_all_gathers = True
    trainer.accelerator.state.fsdp_plugin.sync_module_states = False

    from torch.distributed.fsdp.fully_sharded_data_parallel import BackwardPrefetch
    trainer.accelerator.state.fsdp_plugin.backward_prefetch = BackwardPrefetch.BACKWARD_PRE

    # Identify which modules have "_fsdp_wrap" attribute set to True and wrap these
    def fsdp_policy_fn(module):
        return getattr(module, "_fsdp_wrap", False)

    # Identify which modules have "layer" in their class name and use these
    # as the basic FSDP blocks that are sharded and exchanged between GPUs
    # def layer_policy_fn(module):
        # return "layer" in module.__class__.__name__.lower()

    auto_wrap_policy = functools.partial(lambda_auto_wrap_policy,
                                            lambda_fn=fsdp_policy_fn)
    trainer.accelerator.state.fsdp_plugin.auto_wrap_policy = auto_wrap_policy
    # trainer.accelerator.state.fsdp_plugin.use_orig_params = True



if training_args.do_train:
    checkpoint = None
    print("start training...")
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload
    # trainer.save_tokenizer()

    metrics = train_result.metrics

    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()