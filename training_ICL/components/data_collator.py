# coding=utf-8
'''Preprocess llama pre-training data'''

import numpy as np
import torch
import datasets
from datasets import load_dataset, load_from_disk, DatasetDict, concatenate_datasets
from transformers import DataCollatorWithPadding
import evaluate
from transformers import set_seed, BertTokenizer
from typing import List, Dict, Any, Optional
from functools import partial

from IPython import embed


class DataCollator_masked_doc:

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        encoded_inputs = {key: [example[key] for example in features] for key in features[0].keys()}
        batch = {}
        input_ids = []
        labels = []

        for example, label in zip(encoded_inputs['input_ids'], encoded_inputs['labels']):
            label.append(self.tokenizer.eos_token_id)
            inputs = np.concatenate([example, label], axis=0)
            label = np.asarray([-100] * len(example) + list(label))
            input_ids += [inputs]
            labels += [label]

        padding_tokenid = self.tokenizer.eos_token_id
        inputs_max_length = max(len(x) for x in input_ids)
        labels_max_length = max(len(x) for x in labels)

        padded_inputs = np.ones((len(input_ids), inputs_max_length), dtype=int) * padding_tokenid
        padded_label = np.ones((len(input_ids), labels_max_length), dtype=int) * padding_tokenid
        padded_attention = np.zeros_like(padded_label)

        for i, (inputs, label) in enumerate(zip(input_ids, labels)):
            padded_inputs[i, :len(inputs)] = inputs
            padded_label[i, :len(label)] = label
            padded_attention[i, :len(label)] = 1

        batch['input_ids'] = torch.tensor(padded_inputs)
        batch['labels'] = torch.tensor(padded_label)
        batch['attention_mask'] = torch.tensor(padded_attention)
        return batch


class DataCollator_unmasked:

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        encoded_inputs = {key: [example[key] for example in features] for key in features[0].keys()}
        batch = {}
        input_ids = []
        labels = []

        for example, label in zip(encoded_inputs['input_ids'], encoded_inputs['labels']):
            inputs = np.concatenate([example, label], axis=0)
            label = np.copy(inputs)
            label.append(self.tokenizer.eos_token_id)
            input_ids += [inputs]
            labels += [label]
        
        padding_tokenid = self.tokenizer.eos_token_id
        inputs_max_length = max(len(x) for x in input_ids)
        labels_max_length = max(len(x) for x in labels)

        padded_inputs = np.ones((len(input_ids), inputs_max_length), dtype=int) * padding_tokenid
        padded_label = np.ones((len(input_ids), labels_max_length), dtype=int) * padding_tokenid
        padded_attention = np.zeros_like(padded_label)

        for i, (inputs, label) in enumerate(zip(input_ids, labels)):
            padded_inputs[i, :len(inputs)] = inputs
            padded_label[i, :len(label)] = label
            padded_attention[i, :len(label)] = 1

        batch['input_ids'] = torch.tensor(padded_inputs)
        batch['labels'] = torch.tensor(padded_label)
        batch["attention_mask"] = torch.tensor(padded_attention)

        # batch['input_ids'] = torch.tensor(input_ids)
        # batch['labels'] = torch.tensor(labels)
        # batch["attention_mask"] = torch.ones_like(batch["labels"])
        return batch

