import torch
import numpy as np
import torch.nn as nn

import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import sys
import os

D = 10
L = 1000
N = 10
temp = 25

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

translations = []
for d in range(D):
    P = np.random.permutation(np.eye(N**2))
    translations.append(torch.tensor(P).float().to(device))

shifted_inds = [L-1] + list(range(0, L-1))
shift_mat = torch.zeros(L, L)
for i in range(L):
    shift_mat[i, shifted_inds[i]] = 1
shift_mat = shift_mat.unsqueeze(0).to(device)

I_n = torch.eye(N)
Ones_n = torch.ones(N, 1)
P1 = torch.kron(I_n, Ones_n)
P2 = torch.kron(Ones_n, I_n)
Q = (P1 @ P2.T).unsqueeze(0).to(device)
Q_t = Q.transpose(1, 2)
vocab = [chr(i) for i in range(97, 97 + N)]

def shift(input_mat):
    shifted_input_mat = torch.matmul(input_mat, shift_mat)
    left = Q @ input_mat
    right = Q_t @ shifted_input_mat
    return left * right

def MLT(input_mat, dictionaries):

    ret = input_mat
    # print(ret)
    for D in dictionaries:
        # print(decode(ret[0]))
        ret = D @ ret
        ret = shift(ret)
        # decode_dict(D)
        # print(decode(ret[0]))
    return ret

def create_mask(gt_dict, mask_inds):
    mask = gt_dict.clone()
    for i in range(len(mask_inds)):
        mask[:, mask_inds[i]] = 0
    return mask

def collect_grad(input_batch, label_batch, weights, masked_entries):

    assert len(masked_entries) == D

    P_mats = []
    for i in range(D):
        mask = create_mask(translations[i], masked_entries[i])
        P_mats.append(F.softmax(temp * (weights[i] + mask), dim=0))
    
    output = MLT(input_batch, P_mats)

    output_flatten = torch.cat([x for x in output], dim=1).T
    label_flatten = torch.cat([x for x in label_batch], dim=1).T

    # loss = torch.mean((output - label_batch) ** 2)
    loss = F.cross_entropy(torch.log(output_flatten + 1e-10), torch.argmax(label_flatten, dim=1))
    # print(loss.item())
    loss.backward()

    grad_mat = weights[0].grad.detach().clone()
    weights[0].grad = None
    return grad_mat

def mat_grad_normalizing(grad_mat, gt_mat):
    signal_mat = grad_mat * gt_mat
    noise_mat = grad_mat * (1 - gt_mat)

    signal_values = torch.sum(signal_mat, dim=0, keepdim=True)
    noise_values = torch.sum(noise_mat, dim=0, keepdim=True) / len(gt_mat[0] - 1)
    noise_mean = torch.mean(noise_values, dim=0, keepdim=True)
    noise_min = torch.min(noise_values + gt_mat*100, dim=0, keepdim=True).values

    accuracy = (signal_values < noise_min).float()

    scale = (signal_values - noise_mean) - 1e-8
    rescaled_grad_mat = (grad_mat - noise_mean) / scale
    return rescaled_grad_mat, accuracy

def grad_processing(grad_mat, mask_inds, gt_mat):
    
    mask_entries = mask_inds[0]
    rescaled_mat, accuracy = mat_grad_normalizing(grad_mat, gt_mat)
    noise_entries = []
    acc_ls = []
    for i in mask_entries:
        gt_entry = torch.argmax(gt_mat[:, i])
        acc_ls.append(accuracy[0,i].item())
        noise_entries.append(torch.concat([rescaled_mat[:, i][:gt_entry], rescaled_mat[:, i][gt_entry+1:]]))
    noise_entries = torch.cat(noise_entries, dim=0)

    return noise_entries, acc_ls

num_mask_inds = int(sys.argv[1])
n_batches = 500
print(f'num_mask_inds: {num_mask_inds}', flush=True)
batchsize = 10

layers_drop = list(range(1, D + 1))

stds = []
accs = []
means = []

for n_layers_drop in tqdm(layers_drop):

    print(f'num_mask_inds: {num_mask_inds}', flush=True)

    weights = []
    for i in range(D):
        weights.append(torch.zeros_like(translations[0]))
    weights[0].requires_grad = True

    noise_entries_total = []
    acc_ls_total = []
    for i in range(n_batches):
        random_seq = np.random.choice(N**2, (batchsize, L))
        input_batch = (F.one_hot(torch.tensor(random_seq), N**2).float().transpose(-1, -2)).to(device)
        with torch.no_grad():
            label_batch = MLT(input_batch, translations[:D])
            
        mask_inds = [list(np.random.choice(N**2, num_mask_inds, replace=False))] * n_layers_drop + [[]] * (D - n_layers_drop)
        
        grad_mat = collect_grad(input_batch, label_batch, weights, mask_inds)
        noise_entries, accuracy_ls = grad_processing(grad_mat, mask_inds, translations[0])
        noise_entries_total.append(noise_entries)
        acc_ls_total += accuracy_ls
        
    noise_entries_total = torch.cat(noise_entries_total, dim=0)
    stds.append(torch.std(noise_entries_total).item())
    means.append(torch.mean(noise_entries_total).item())
    accs.append(np.mean(acc_ls_total))

meta = {
    'stds': stds,
    'means': means,
    'accs': accs,
    'mask_size': num_mask_inds,
    'batchsize': batchsize,
    'n_batches': n_batches,
    'layers_drop': layers_drop
}

os.makedirs(f'./logs/var_nlayer/N{N}D{D}L{L}T{temp}', exist_ok=True)
json.dump(meta, open(f'./logs/var_nlayer/N{N}D{D}L{L}T{temp}/masksize{num_mask_inds}.json', 'w'))