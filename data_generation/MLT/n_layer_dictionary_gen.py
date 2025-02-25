"""
This script generates a multi-layer translation dictionary and uses it to generate source-target sequence pairs for training and evaluation.
Functions:
    create_vocab_chars(n_chars, n_layers):
        Creates characters for each layer of vocabulary.
    create_vocab_tuples(vocab_chars, tuple_lengths):
        Creates vocabulary tuples for each layer of vocabulary.
    generate_random_translation(tuples_1, tuples_2):
        Creates a translation dictionary between two vocabularies.
    random_translations(vocab_tuples):
        Generates random translations between multiple layers of vocabulary.
    tokens2str(tokens):
        Converts a list of tokens to a string.
    translate_lookahead(tokens, translation):
        Translates the longest matching sequence using a lookahead mechanism.
    generate_source(source_tuples, n_tuples, transition_matrix, init_vectors):
        Generates source sequences without held-out vocabulary using rejection sampling.
    sample_held_out_vocab(source_tuples, held_out_ratio):
        Samples held-out vocabulary from the source vocabulary without any confounding overlap.
Classes:
    Dictionary:
        A class to create and manage multi-layer translation dictionaries.
        Methods:
            __init__(self, chars_size, n_translation, tuples_size, held_out_ratio=0.0, seed=42):
                Initializes the Dictionary object.
            vocab_meta(self):
                Returns metadata about the vocabulary.
            generate_source_target_lookahead(self, n_tuples, transition_matrix=None, init_vectors=None):
                Generates source-target sequence pairs using lookahead translation.
            batch_generate_lookahead(self, n_samples, length_range, ds_path, seq_seed, transition_matrix=None, init_vectors=None, n_threads=1):
                Generates batches of source-target sequence pairs using lookahead translation.
Main:
    The script can be run as a standalone program to generate dictionaries and datasets based on command-line arguments.
"""

import numpy as np
import pickle
import json
import os
import argparse
import torch
from tqdm import tqdm, trange
from IPython import embed

from joblib import Parallel, delayed

# create characters for each layer of vocabulary
def create_vocab_chars(n_chars, n_layers):
    return [np.arange(n_chars) for _ in range(n_layers)]

# create vocabulary tuples for each layer of vocabulary
def create_vocab_tuples(vocab_chars, tuple_lengths):
    vocab_tuples = []
    for chars in vocab_chars:
        tuples = set()
        for l, n in enumerate(tuple_lengths):
            l_tuples = set()
            while len(l_tuples) < n:
                l_tuples.add(tuple(np.random.choice(chars, size=l + 1, replace=True).tolist()))
            tuples.update(l_tuples)
        vocab_tuples.append(list(tuples))
    return vocab_tuples

# create translation dictionary between two vocabularies
def generate_random_translation(tuples_1, tuples_2):
    mapping = np.random.choice(np.arange(len(tuples_2)), size=len(tuples_1), replace=False)
    return {tuples_1[i]: tuples_2[mapping[i]] for i in range(len(tuples_1))}

# generate random translations between multiple layers of vocabulary
def random_translations(vocab_tuples):
    translations = []
    for i in range(len(vocab_tuples) - 1):
        translations.append(generate_random_translation(vocab_tuples[i], vocab_tuples[i + 1]))
    return translations

def tokens2str(tokens):
    return '.'.join([str(c) for c in tokens])

def translate_lookahead(tokens, translation):
    # lookahead translation mechanism: translate the longest matching sequence so far
    # returns the input (parsed) and output (parsed)
    parsing_result = []
    ret = []
    max_tuple_len = max([len(t) for t in translation.keys()])

    remaining = tokens
    while len(remaining) > 0:
        t_match = max_tuple_len
        while t_match > 0:
            potential_match = tuple(remaining[:t_match])
            # print(potential_match)
            if potential_match in translation.keys():
                parsing_result.append(potential_match)
                ret.append(translation[potential_match])
                remaining = remaining[t_match:]
                break
            t_match -= 1
        if t_match == 0:
            raise AssertionError("Token sequence {}\n cannot be translated by dictionary {}\n".format(remaining, translation))
    
    return parsing_result, ret

# rejection sampling method for sampling source sequences without held out vocabulary
# since we are only using 2-tuples, we can do not need to worry about overlapping substrings
def generate_source(source_tuples, n_tuples, transition_matrix, init_vectors):

    vocab_inds = np.arange(len(source_tuples))
    assert transition_matrix.shape == (len(source_tuples), len(source_tuples))
    assert np.allclose(np.sum(transition_matrix, axis=1), 1.0, 1e-5)
    assert np.isclose(np.sum(init_vectors), 1.0, 1e-5)

    # Sample first token tuple
    gen_inds = [np.random.choice(vocab_inds, p=init_vectors)]
    gen_seqs = list(source_tuples[gen_inds[-1]])

    # Start generating
    for _ in range(n_tuples - 1):
        next_ind = np.random.choice(vocab_inds, p=transition_matrix[gen_inds[-1]])
        next_w = list(source_tuples[next_ind])
        gen_inds.append(next_ind)
        gen_seqs += next_w
    
    return gen_seqs, gen_inds

# sample held out vocabulary from the source vocabulary without any confounding overlap
# select all strings that are not substrings of any other strings
def sample_held_out_vocab(source_tuples, held_out_ratio):
    unique_words = []
    strings_reps = [tokens2str(x) for x in source_tuples]
    for word in source_tuples:
        word_str = tokens2str(word)
        if all(word_str not in s for s in strings_reps if len(s) > len(word_str)):
            unique_words.append(word)

    held_out_size = int(len(unique_words) * held_out_ratio)

    if held_out_size > len(unique_words):
        raise AssertionError("The number of unique words is smaller than the held out population")

    held_out_tuples_inds = np.random.choice(np.arange(len(unique_words)), size=held_out_size, replace=False)
    held_out_tuples = [unique_words[i] for i in held_out_tuples_inds]
    
    remaining_tuples = []
    for x in source_tuples:
        if x not in held_out_tuples:
            remaining_tuples.append(x)

    return remaining_tuples, held_out_tuples


class Dictionary():

    def __init__(self, chars_size, n_translation, tuples_size, held_out_ratio=0.0, seed=42):

        np.random.seed(seed)
        self.seed = seed
        self.n_translation = n_translation
        self.chars_sets = create_vocab_chars(chars_size, n_translation + 1)
        self.words_sets = create_vocab_tuples(self.chars_sets, tuples_size)
        self.dictionaries = random_translations(self.words_sets)
        self.remain_vocab, self.held_out_vocab = sample_held_out_vocab(self.words_sets[0], held_out_ratio)

    @property
    def vocab_meta(self):
        return {
            'chars': self.chars_sets,
            'words': self.words_sets,
            'dictionaries': self.dictionaries,
            'remain_vocab': self.remain_vocab,
            'held_out_vocab': self.held_out_vocab,
            'dict_seed': self.seed
        }
    
    def generate_source_target_lookahead(self, n_tuples, transition_matrix=None, init_vectors=None):

        raw_source_shifted, raw_inds = generate_source(self.remain_vocab, n_tuples, transition_matrix, init_vectors)
        raw_source = [raw_source_shifted[-1]] + raw_source_shifted[:-1] # preset the offset to make the translation heldout
        tmp_source = raw_source.copy()
        parsed_sources = []

        for dictionary in self.dictionaries:
            tmp_source = tmp_source[1:] + [tmp_source[0]] # shift the first token to the last token (bitshift)
            tmp_source_parsed, target_parsed = translate_lookahead(tmp_source, dictionary)
            target_raw = []
            for x in target_parsed:
                target_raw += x
            tmp_source = target_raw
            parsed_sources.append(tmp_source_parsed)
        
        raw_target = []
        for x in target_parsed:
            raw_target += x

        assert len(raw_source) == len(raw_target), "target seq should have the same length as the source"
        return raw_source, raw_target, parsed_sources
    
    def batch_generate_lookahead(self, n_samples, length_range, ds_path, seq_seed, transition_matrix=None, init_vectors=None, n_threads=1):

        assert seq_seed is not None
        seq_seed = seq_seed * n_threads # each thread has a different seed
        N = len(self.remain_vocab)
        if transition_matrix is None:
            transition_matrix = np.ones((N, N)) / N
        if init_vectors is None:
            init_vectors = np.ones(N) / N
        np.random.seed(seq_seed)

        meta = self.vocab_meta
        gen_meta = {
            'transition_matrix': transition_matrix,
            'init_vectors': init_vectors,
            'n_samples': n_samples,
            'length_range': length_range,
            'seq_seed': seq_seed
        }
        meta.update(gen_meta)
        
        def gen_batch_parallel(n_samples, thread_seed):
            np.random.seed(thread_seed)
            batch_ret = []
            for _ in range(n_samples):
                n_tuples = np.random.choice(length_range)
                source, target, parsed_sources = self.generate_source_target_lookahead(n_tuples, transition_matrix, init_vectors)
                batch_ret.append([source, target, parsed_sources])
            return batch_ret
        
        if n_threads == 1:
            translate_seqs = gen_batch_parallel(n_samples, seq_seed)
        
        else:
            thread_jobs_desc = [(n_samples // n_threads + 1, seq_seed + i) for i in range(n_threads)]
            batch_rets = Parallel(n_jobs=n_threads, verbose=50)(delayed(gen_batch_parallel)(*job) for job in thread_jobs_desc)
            translate_seqs = []
            for ret in batch_rets:
                translate_seqs += ret

        if ds_path is not None:
            os.makedirs(ds_path, exist_ok=True)        
            pickle.dump(meta, open(os.path.join(ds_path, 'meta.pkl'), 'wb'))
            pickle.dump(translate_seqs, open(os.path.join(ds_path, 'raw_ds.pkl'), 'wb'))
        return meta, translate_seqs

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Path to the raw dataset", type=str, default='')
    parser.add_argument("--attr", type=str, default="ICLPT", help='ICLPT (For ICL training) or IM (for learning a specific set of phrasebooks)')

    parser.add_argument("--nchars", type=int, default=10, help='number of characters in the vocabulary (n in the paper)')
    parser.add_argument("--depth", type=int, default=4, help='depth of translations (d in the paper)')
    parser.add_argument("--n_dictionaries", type=int, default=20000, help='number of dictionaries to generate')
    parser.add_argument("--n_samples_per_dictionary", type=int, default=100, help='number of samples per dictionary')
    
    parser.add_argument("--length_min", type=int, default=10, help='min length of sequences (in tuples)')
    parser.add_argument("--length_max", type=int, default=20, help='max length of sequences (in tuples)')

    parser.add_argument("--heldout_ratio", type=float, default=0.0, help='heldout ratio')
    parser.add_argument("--n_eval", type=int, default=1024, help='number of evaluation samples')
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    
    tuple_sizes = [0, args.nchars ** 2] # Only generate 2-tuples from the vocabulary
    ds_name = f'{args.attr}_c{args.nchars}_D{args.depth}_nD{args.n_dictionaries}_N{args.n_samples_per_dictionary}_S{args.seed}_LCont{args.length_min}-{args.length_max}/HO{args.heldout_ratio}'
    print(ds_name, flush=True)

    if args.attr == 'ICLPT':
        n_dict = args.n_dictionaries + args.n_eval
        n_samples_per_dictionary = args.n_samples_per_dictionary
    else:
        n_dict = args.n_dictionaries
        n_samples_per_dictionary = args.n_samples_per_dictionary + args.n_eval
    print(f"Generating {n_dict} dictionaries with {n_samples_per_dictionary} samples per dictionary", flush=True)

    def gen_metas(i, n_threads=1):
        dictionary = Dictionary(args.nchars, args.depth, tuple_sizes, args.heldout_ratio, args.seed + i)
        gen_meta, raw_ds = dictionary.batch_generate_lookahead(n_samples_per_dictionary, np.arange(args.length_min, args.length_max + 1), ds_path=None, seq_seed=args.seed + i, n_threads=n_threads)
        return [gen_meta, raw_ds]

    if args.n_dictionaries == 1:
        dictionaries_meta = [gen_metas(0, n_threads=32)]
    else:
        dictionaries_meta = Parallel(n_jobs=32, verbose=50)(delayed(gen_metas)(i) for i in range(n_dict))

    ds_path = os.path.join(args.path, ds_name)
    os.makedirs(ds_path, exist_ok=True)
    pickle.dump(dictionaries_meta, open(os.path.join(ds_path, 'dictionaries_meta.pkl'), 'wb'))


    # If we are generating the dataset for ICLPT, we need to split the dataset into train and test by splitting the set of phrasebooks
    if args.attr == 'ICLPT':
        print("Splitting the dataset for ICLPT")
        train_split = dictionaries_meta[:-args.n_eval]
        test_split = dictionaries_meta[-args.n_eval:]
        for split_size in [10 ** i for i in range(1, np.log10(len(dictionaries_meta) - 1).astype(int) + 2)]:
            for split_name in ['train', 'test']:
                split_path = os.path.join(ds_path, f'{split_name}_{split_size}.pkl')
                if split_name == 'train':
                    split = train_split[:split_size]
                else:
                    split = test_split[:split_size]
                    if split_size >= len(test_split) * 10:
                        continue
                pickle.dump(split, open(split_path, 'wb'))

    # If we are generating the dataset for IM, we need to split the dataset into train and test by splitting the set of examples for the same set of phrasebooks    
    else:
        assert args.attr == 'IM'
        print("Splitting the dataset for IM")
        n_samples = len(dictionaries_meta[0][1])
        train_split = [[dictionaries_meta[0][0], dictionaries_meta[0][1][:-args.n_eval]]]
        test_split = [[dictionaries_meta[0][0], dictionaries_meta[0][1][-args.n_eval:]]]

        for split_size in [10 ** i for i in range(1, np.log10(n_samples - 1).astype(int) + 2)]:
            for split_name in ['train', 'test']:
                split_path = os.path.join(ds_path, f'{split_name}_{split_size}.pkl')
                if split_name == 'train':
                    split = [[train_split[0][0], train_split[0][1][:split_size]]]
                else:
                    split = [[test_split[0][0], test_split[0][1][:split_size]]]
                    if split_size >= len(test_split[0][1]) * 10:
                        continue
                pickle.dump(split, open(split_path, 'wb'))
