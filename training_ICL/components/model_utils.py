import torch
import os
import json

from IPython import embed

def add_new_tokens(model, tokenizer, new_tokens, update_embedding=True):
    """Add a new token to the model and tokenizer."""

    total_padding = 64
    n_added_tokens = 0
    for token in new_tokens:
        if token not in tokenizer.get_vocab():
            tokenizer.add_tokens([token])
            n_added_tokens += 1
            print(f"Added token '{token}' to the tokenizer.")
        else:
            print(f"Token '{token}' already exists in the tokenizer.")
    
    # dummy_tokens = [f"<dummy_padding_{i}>" for i in range(total_padding - n_added_tokens)]
    # tokenizer.add_tokens(dummy_tokens)
    if not update_embedding:
        print("Not updating the model's embedding")
        return
    
    old_embeddings = model.get_input_embeddings().weight.clone()
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=total_padding)
    print(f"Resized token embeddings from {old_embeddings.size()} to {model.get_input_embeddings().weight.size()}.")
    new_embeddings = model.get_input_embeddings().weight
    new_embeddings.data[:old_embeddings.size(0)] = old_embeddings
    new_embeddings.data[-total_padding:] = torch.normal(mean=0.0, std=0.02, size=new_embeddings.data[-n_added_tokens:].size())


def archive_datasets(train_set, test_sets, tokenizer, output_dir, n_samples=20):

    subsets = {}
    print("Archiving dataset samples...", flush=True)
    print("train_set:", len(train_set), flush=True)
    print("test_sets:", {k: len(test_sets[k]) for k in test_sets}, flush=True)
    if n_samples != -1:
        subsets['train'] = [train_set[i] for i in range(n_samples)]
        for k in test_sets:
            subsets[k] = [test_sets[k][i] for i in range(n_samples)]
    else:
        n_samples = len(train_set)
        subsets['train'] = [train_set[i] for i in range(n_samples)]
        for k in test_sets:
            n_samples = len(test_sets[k])
            subsets[k] = [test_sets[k][i] for i in range(n_samples)]

    decoded_subsets = {}    
    for k in subsets:
        # decode all samples
        decoded_samples = []
        for sample in subsets[k]:
            decoded_input_ids = tokenizer.decode(sample['input_ids'])
            decoded_labels = tokenizer.decode(sample['labels'])
            decoded_samples.append({'input': decoded_input_ids, 'labels': decoded_labels})
        decoded_subsets[k] = decoded_samples

    json.dump(decoded_subsets, open(os.path.join(output_dir, 'ds_sample.json'), 'w'), indent=4)
    print("Dataset archived to", os.path.join(output_dir, 'ds_sample.json'), flush=True)
    return

