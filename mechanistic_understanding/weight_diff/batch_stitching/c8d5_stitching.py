from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import json
import os

import numpy as np
import sys

import matplotlib.pyplot as plt
from tqdm import tqdm

import json

import pickle

ICL_path = # PATH TO ICL MODEL
IM_path_template = # PATH TO IM MODEL (with model_mask as a format string)
IM_full_path = sys.argv[1]
IM_full_model = AutoModelForCausalLM.from_pretrained(IM_full_path, device_map='auto', torch_dtype=torch.float16)


ICL_model = AutoModelForCausalLM.from_pretrained(ICL_path, device_map='auto', torch_dtype=torch.float16)
MERGE_model = AutoModelForCausalLM.from_pretrained(ICL_path, device_map='auto', torch_dtype=torch.float16)

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B-Instruct')
def get_breakpoint(tokens, sep_tokens):
    break_point = 0
    for i in range(len(sep_tokens) + 1, len(tokens[0])):
        if torch.all(tokens[0][-i: -i+len(sep_tokens)] == sep_tokens):
            break_point = -i + len(sep_tokens)
            break
    return break_point

def get_acc_loss(model, tokens, sep_tokens):
    sep_point = get_breakpoint(tokens, sep_tokens)

    with torch.no_grad():
        output = model(tokens, return_dict=True)
        logits_localized = output.logits[:, (sep_point - 1): -3]
        preds_localized = logits_localized[0].argmax(dim=-1)
        gt_localized = tokens[:, (sep_point): -2]

        matching_acc = (preds_localized == gt_localized)[0].float()
        matching_loss = F.cross_entropy(logits_localized[0], gt_localized[0], reduction='none')
    return matching_acc, matching_loss

def compute_metrics(model, eval_seqs, sep_tokens):
    matching_accs = []
    matching_losses = []
    for tokens in eval_seqs:
        accs, losses = get_acc_loss(model, tokens.to(model.device), sep_tokens.to(model.device))
        matching_accs.extend(accs.cpu().numpy().tolist())
        matching_losses.extend(losses.cpu().numpy().tolist())
    return np.mean(matching_accs), np.mean(matching_losses)

def merge_model_by_layers(model_base, model_new, model_merge, target_layers):
    base_sd = model_base.state_dict()
    new_sd = model_new.state_dict()
    for layer_ind in range(len(model_merge.model.layers)):
        if layer_ind in target_layers:
            model_merge.model.layers[layer_ind] = model_new.model.layers[layer_ind]
        else:
            model_merge.model.layers[layer_ind] = model_base.model.layers[layer_ind]
    return model_merge


model_masks = [
    '1-0-0-0-0',
    '0-1-0-0-0',
    '0-0-1-0-0',
    '0-0-0-1-0',
    '0-0-0-0-1',
]


ret = {}

for mask in model_masks:

    print(mask)
    IM_path = IM_path_template.format(MASK=mask)
    ds_sample = json.load(open(IM_path + "/ds_sample.json"))
    data_key = 'IM_test_DMaskon'

    eval_seqs = []
    for x in ds_sample[data_key]:
        eval_seqs.append(tokenizer.encode(x['input'] + x['labels'], return_tensors='pt'))

    sep_str = " Sequence in language 6:"
    sep_tokens = tokenizer.encode(sep_str, return_tensors='pt').to(ICL_model.device)[0][1:]

    n_layers = len(ICL_model.model.layers)
    acc_mat = np.zeros((n_layers, n_layers))
    loss_mat = np.zeros((n_layers, n_layers))

    fixed_layer = 0

    for last_layer in tqdm(range(fixed_layer, n_layers, 1)):
        for first_layer in range(fixed_layer, last_layer + 1, 1):
            MERGE_model = merge_model_by_layers(ICL_model, IM_full_model, MERGE_model, list(range(fixed_layer)) + list(range(first_layer, last_layer + 1)))
            acc, loss = compute_metrics(MERGE_model, eval_seqs[:20], sep_tokens)
            acc_mat[first_layer, last_layer] = acc
            loss_mat[first_layer, last_layer] = loss
    
    ret[mask] = {
        'acc': acc_mat,
        'loss': loss_mat
    }

save_folder = './results/' + (IM_full_path.split('/')[-2])
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

save_dest = './results/' + '/'.join(IM_full_path.split('/')[-2:])
pickle.dump(ret, open(save_dest + '.pickle', 'wb'))



fig = plt.figure(figsize=(15, 2.8), dpi=300)
gs = fig.add_gridspec(1, len(model_masks) + 1, width_ratios=[1] * len(model_masks) + [0.075], wspace=0.2)  # Extra space for the colorbar
# gs = fig.add_gridspec(1, len(model_masks) + 1, width_ratios=[1] * len(model_masks) + [0.075])  # Extra space for the colorbar

names = [r"Dropping STR($\pi^*_{i}$)".format(i=i) for i in range(1, 6)]
ticks_density = 3

for i, mask in enumerate(model_masks):
    ax = fig.add_subplot(gs[0, i])
    ax.set_title(names[i])


    mat = ret[mask]['acc'][fixed_layer:, fixed_layer:]
    im = ax.imshow(mat, cmap='hot', interpolation='nearest', vmin=0, vmax=1, origin='lower')
    ax.grid(True, alpha=0.35)
    ax.set_xticks(np.arange(0, n_layers - fixed_layer, ticks_density))
    ax.set_xticklabels(np.arange(1 + fixed_layer, 1 + n_layers, ticks_density))
    ax.set_yticks(np.arange(0, n_layers - fixed_layer, ticks_density))
    ax.set_yticklabels(np.arange(1 + fixed_layer, 1 + n_layers, ticks_density))
    if i == 0:
        ax.set_ylabel("$L_{start}$", fontsize=14)
    ax.set_xlabel("$L_{end}$", fontsize=14)

# Add a colorbar in the last gridspec column
cbar_ax = fig.add_subplot(gs[0, -1])
fig.colorbar(im, cax=cbar_ax, orientation='vertical', label='Evaluation Accuracy', pad=0.0, shrink=0.1)
plt.tight_layout()
plt.savefig(save_dest + ".pdf", bbox_inches='tight', pad_inches=0)
plt.show()