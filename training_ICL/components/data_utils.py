from tqdm import tqdm
import os
import pickle
from datasets import Dataset
import numpy as np
import copy

from joblib import Parallel, delayed

from .prompts import TEMPLATE
from IPython import embed


def llama3_chat_template(message):
    return f"<|begin_of_text|><|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


def encode_string(ids, chars):
    ret = ''
    for i in ids:
        new_char = chars[i]
        if new_char.startswith('<|') and new_char.endswith('|>'):
            ret += new_char
        else:
            ret += ' ' + new_char
    return ret


def filter_dictionary(dictionary, parsed_source, irrelevant_rate=0):
    """
    Filter the dictionary based on the parsed source and irrelevant rate
    """
    assert irrelevant_rate >= 0 and irrelevant_rate <= 1, "Invalid irrelevant rate, should be in [0, 1]"

    parsed_sources_dedup = set(parsed_source)
    keys = set(dictionary.keys())
    irrelevant_keys = list(keys - parsed_sources_dedup)

    irrelevant_inds = np.arange(len(irrelevant_keys))
    np.random.shuffle(irrelevant_inds)
    n_irrelevant = int(len(irrelevant_inds) * irrelevant_rate)

    total_keys = list(parsed_sources_dedup) + irrelevant_keys[:n_irrelevant]
    np.random.shuffle(total_keys)
    
    filtered_dict_shuffled = {}
    for k in total_keys:
        filtered_dict_shuffled[k] = dictionary[k]

    return filtered_dict_shuffled


def encode_dictionary_with_mask(dictionary, mask_rate, source_chars, target_chars, dictionary_index, curriculum_args):
    """
    Encode the dictionary with masking
    """
    ret = []
    keys = list(dictionary.keys())
    key_inds = np.arange(len(keys))

    masked_inds = np.random.choice(key_inds, int(len(keys) * mask_rate), replace=False)
    masked_keys = [keys[i] for i in masked_inds]

    for k in dictionary:
        
        source_encoded = encode_string(k, source_chars)
        dest_encoded = encode_string(dictionary[k], target_chars)
        
        if k in masked_keys:
            # If masking token is set to 'remove', skip the dictionary entry
            if 'remove' in curriculum_args.dictionary_mask_token_template:
                continue

            mask_token = curriculum_args.dictionary_mask_token_template.format(index=curriculum_args.dictionary_mask_token_base_index + dictionary_index)
            source_encoded = mask_token * len(k)
            dest_encoded = mask_token * len(dictionary[k])
        
        ret.append(TEMPLATE.mapping_template.format(
            tuple_source=source_encoded,
            tuple_dest=dest_encoded
        ))

    return ''.join(ret)


def encode_dictionaries(
        dictionaries,
        sample,
        lang_names,
        chars_list,
        format_config,
        task_name,
        data_args,
        curriculum_args,
    ):

    progressive_mask_rates = format_config['dict_mask_rates']
    dictionary_filter_rates = format_config['dict_filter_rates']
    assert len(progressive_mask_rates) == len(dictionaries)

    dictionaries_filtered = []
    for i, dictionary in enumerate(dictionaries):
        parsed_sources = sample[2][i]
        filtered_dict = filter_dictionary(dictionary, parsed_sources, dictionary_filter_rates[i])
        dictionaries_filtered.append(filtered_dict)

    dictionaries_desc = TEMPLATE.dictionaries_header_subset
    dictionaries_indices = np.arange(len(dictionaries_filtered))
    
    if data_args.dict_permutation:
        np.random.shuffle(dictionaries_indices)

    # Encode the dictionaries subject to the masks
    for i in dictionaries_indices:
        dictionary = dictionaries_filtered[i]
        if curriculum_args.dictionary_mask_indices == 'same':
            dictionary_index = 0
        elif curriculum_args.dictionary_mask_indices == 'different':
            task_base_index = int(task_name.split('_')[-1]) * len(dictionaries_indices)          #<-- to denote unique per dictionary in multi-dictionary experiments.
            dictionary_index = i + task_base_index
        else:
            dictionary_mask_indices = curriculum_args.dictionary_mask_indices
            dictionary_mask_indices = dictionary_mask_indices.split('-')
            assert len(dictionary_mask_indices) == len(dictionaries_indices)
            dictionary_index = int(dictionary_mask_indices[i])

        dictionary_encoded = encode_dictionary_with_mask(
            dictionary,
            progressive_mask_rates[i],
            chars_list[i],
            chars_list[i + 1],
            dictionary_index, # TODO: Change later if wanting to specify the dictionary index
            curriculum_args
        )
        dictionaries_desc += TEMPLATE.dictionary_desc.format(
            lang_in=lang_names[i],
            lang_out=lang_names[i + 1],
            dictionary=dictionary_encoded
        )
    return dictionaries_desc


def add_thinking_tokens(token_seq, think_rate, think_mode):
    """
    Format the token sequence with some tokens substituted to thinking tokens
    """
    think_token_id = -1
    assert think_rate >= 0 and think_rate <= 1, "Invalid think rate, should be in [0, 1]"

    n_think_tokens = int(len(token_seq) * think_rate)
    think_inds = np.arange(len(token_seq))

    if think_mode == 'random':
        np.random.shuffle(think_inds)
        for i in think_inds[:n_think_tokens]:
            token_seq[i] = think_token_id
    elif think_mode == 'forward':
        for i in range(n_think_tokens):
            token_seq[i] = think_token_id
    elif think_mode == 'backward':
        for i in range(n_think_tokens):
            token_seq[-i-1] = think_token_id
    else:
        raise NotImplementedError("Invalid think mode")

    return token_seq


def format_cot_answer(
        sample,
        dictionaries,
        char_list,
        format_config,
        data_args,
        curriculum_args
    ):

    think_rates = format_config['think_rates']
    think_mode = curriculum_args.think_token_direction_per_level
    pad_token_id = -2

    assert len(think_rates) == len(dictionaries) - 1, "Invalid think rates"

    # First element in CoT is the question
    q, a, parsed_sources = sample
    cot_ls = [q]

    # Intermediate Steps of CoT
    for i, parsed_source in enumerate(parsed_sources[:-1]):
        cot_token_seq = []
        for x in parsed_source:
            cot_token_seq += dictionaries[i][x]
        assert len(cot_token_seq) <= curriculum_args.a_cot_max_length, "COT answer exceeds the maximum length"

        cot_token_seq += [pad_token_id] * (curriculum_args.a_cot_max_length - len(cot_token_seq))
        cot_token_seq = add_thinking_tokens(cot_token_seq, think_rates[i], think_mode)
        cot_ls.append(cot_token_seq)

    cot_ls.append(a)

    cot_strs = []
    for i, cot in enumerate(cot_ls):
        augmented_char_list = char_list[i] + [curriculum_args.cot_pad_token, curriculum_args.think_token]
        cot_strs.append(encode_string(cot, augmented_char_list))
    assert len(cot_ls) == len(dictionaries) + 1

    ret = []
    for i, cot_str in enumerate(cot_strs):
        ret.append(TEMPLATE.cot_template.format(
            lang=i + 1,
            lang_str=cot_str
        ))
    if not data_args.cot_on:
        ret = [ret[0], ret[-1]]

    return TEMPLATE.str_seperator.join(ret)

def format_cot_demo(d, data_args):
    ret = []
    for k in range(1, d + 2):
        ret.append(TEMPLATE.cot_template.format(
            lang=k,
            lang_str=TEMPLATE.cot_demo_template.format(lang=k)
        ))
    ret[-1] = ret[-1][:-1]
    if not data_args.cot_on:
        ret = [ret[0], ret[-1]]
    return TEMPLATE.str_seperator.join(ret)

def initialize_chars_list(meta, data_args):
    char_count = 0
    chars_list = []
    
    letters = copy.deepcopy(TEMPLATE.abs_letters) if data_args.abs_letters else copy.deepcopy(TEMPLATE.letters)

    if data_args.chars_permutation:
        np.random.shuffle(letters)

    for chars in meta['chars']:
        chars_list.append(letters[char_count:char_count + len(chars)])
        char_count += len(chars)
    
    return chars_list

def format_tutorial_v5(
        meta,
        sample,
        task_name,
        data_args,
        curriculum_args,
        format_config,
    ):
    '''
    This function processes tutorial data to generate input-label pairs for training or evaluation purposes. 

    Parameters:
    - meta: metadata for the tutorial
    - tutorial: list of samples in the tutorial
    - ds_prefix: prefix for the dataset name
    - task_name: name of the task
    - data_args: arguments for data processing
    - tokenizer: tokenizer for encoding the text

    Returns:
    - all_examples: list of input-label pairs for training
    - eval_examples: dictionary of input-label pairs for evaluation
    '''

    chars_list = initialize_chars_list(meta, data_args)
    dictionaries = meta['dictionaries']
    d_translate = len(dictionaries)
    lang_names = TEMPLATE.language_names[:d_translate + 1]

    ### Format context and header
    context_ls = [TEMPLATE.task_header.format(task_name=task_name)]
    q, a, parsed_sources = sample

    ### Format dictionary
    dictionaries_str = encode_dictionaries(
        dictionaries,
        sample,
        lang_names,
        chars_list,
        format_config,
        task_name,
        data_args,
        curriculum_args
    )
    context_ls.append(dictionaries_str)

    ### Format question
    cot_demo = format_cot_demo(d_translate, data_args)
    context_ls.append(TEMPLATE.q_template.format(
        task_name=task_name,
        lang_source=lang_names[0],
        lang_dest=lang_names[-1],
        cot_template_example=cot_demo,
        q_seq=encode_string(q, chars_list[0])
    ))
    context = TEMPLATE.str_seperator.join(context_ls)
    formatted_context = llama3_chat_template(context)
    
    # Generate CoT demo
    cot_a = format_cot_answer(
        sample,
        dictionaries,
        chars_list,
        format_config,
        data_args,
        curriculum_args
    )
    answer = TEMPLATE.a_template_cot.format(cot_answer=cot_a)
    answer += TEMPLATE.eot_token

    return formatted_context, answer

def batch_encode(tokenizer, ds):
    inputs, labels = [], []
    for x in ds:
        inputs.append(x[0])
        labels.append(x[1])
    inputs_encoded = tokenizer.batch_encode_plus(inputs, padding=False, truncation=False)
    labels_encoded = tokenizer.batch_encode_plus(labels, padding=False, truncation=False)
    ret = []
    for i in range(len(inputs)):
        ret.append({'input_ids': inputs_encoded['input_ids'][i], 'labels':labels_encoded['input_ids'][i]})
    return ret

def bin_progress(progress, n_bins, direction='forward'):
    if direction == 'even_disjoint_random':
        rand_choice = np.random.choice([0, 1])
        if rand_choice == 0:
            ret = [progress] * n_bins
        else:
            ret = [0] * n_bins
            rand_ind = np.random.choice(n_bins)
            ret[rand_ind] = progress
        return ret
    elif direction == 'disjoint_random_onehot':
        ret = [0] * n_bins
        rand_ind = np.random.choice(n_bins)
        ret[rand_ind] = progress
        return ret
    
    if progress == 0:
        return [0] * n_bins
    elif progress == 1:
        return [1] * n_bins
    
    if direction in ['forward', 'backward']:
        ret = [0] * n_bins
        for i in range(n_bins):
            if progress >= ((i + 1) / n_bins) - 1e-5:
                ret[i] = 1
            elif progress >= (i / n_bins):
                ret[i] = progress * n_bins - i
            else:
                ret[i] = 0
        if direction == 'backward':
            ret = ret[::-1]

    elif direction == 'even':
        ret = [progress] * n_bins

    elif direction == 'disjoint_random':
        ret = [0] * n_bins
        rand_ind = np.random.choice(n_bins)
        ret[rand_ind] = progress
    else:
        print (direction)
        raise NotImplementedError("Invalid direction")

    return ret


def get_curriculum_progress(x, curriculum):

    if curriculum == 'on':
        return 1
    elif 'on-' in curriculum:
        return float(curriculum.split('-')[1])
    elif curriculum == 'off':
        return 0
    elif 'increase' in curriculum:
        return x
    elif 'increase-' in curriculum:
        return x * float(curriculum.split('-')[1])
    elif '@' in curriculum:
        # curriculum = '0.0@0.3-1.0@0.8'
        
        start_config, end_config = curriculum.split('-')
        start_progress, start_point = start_config.split('@')
        end_progress, end_point = end_config.split('@')
        
        start_progress, end_progress = float(start_progress), float(end_progress)
        start_point, end_point = float(start_point), float(end_point)

        assert start_point < end_point, "Invalid curriculum config"
        assert start_progress < end_progress, "Invalid curriculum config"

        if x < start_point:
            return start_progress
        elif x > end_point:
            return end_progress
        else:
            return start_progress + (end_progress - start_progress) * (x - start_point) / (end_point - start_point)
    #elif 'per-doc' in curriculum:



def generate_format_config(progress, meta, curriculum_args, dict_ip=None, think_rate=None, dict_mask=None):

    dict_ip_curriculum = dict_ip if dict_ip is not None else curriculum_args.dict_irrelevant_rate_curriculum
    think_token_curriculum = think_rate if think_rate is not None else curriculum_args.think_token_curriculum
    dict_mask_curriculum = dict_mask if dict_mask is not None else curriculum_args.dictionary_mask_rate_curriculum
    
    n_dict = len(meta['dictionaries'])
    
    dict_filter_progress = get_curriculum_progress(progress, dict_ip_curriculum)
    dict_filter_rates = bin_progress(dict_filter_progress, n_dict, 'even')

    think_progress = get_curriculum_progress(progress, think_token_curriculum)
    think_rates = bin_progress(think_progress, n_dict - 1, curriculum_args.think_token_direction)
    dict_mask_progress = get_curriculum_progress(progress, dict_mask_curriculum)

    if dict_mask_curriculum == 'off':
        dict_mask_rates = [0] * n_dict
    else:
        dictionary_mask_config = [int(x) for x in curriculum_args.dictionary_mask_config.split('-')]
        assert len(dictionary_mask_config) == n_dict
        active_mapping_locs = []
        for i, x in enumerate(dictionary_mask_config):
            assert x in [0, 1]
            if x == 1:
                active_mapping_locs.append(i)
        dict_mask_rates_raw = bin_progress(dict_mask_progress, sum(dictionary_mask_config), curriculum_args.dictionary_mask_token_direction)
        dict_mask_rates = [0] * n_dict
        for i, x in enumerate(dict_mask_rates_raw):
            dict_mask_rates[active_mapping_locs[i]] = x

    #print (dict_mask_rates)
    format_config = {
        'dict_filter_rates': dict_filter_rates,
        'think_rates': think_rates,
        'dict_mask_rates': dict_mask_rates,
    }
    return format_config

class OnTheFlyDataset(Dataset):
    def __init__(self,
                 raw_dataset,
                 data_args,
                 curriculum_args,
                 tokenizer,
                 dict_ip=None,
                 think_rate=None,
                 dict_mask=None
                ):
        
        self.raw_dataset = raw_dataset
        self.data_args = data_args
        self.curriculum_args = curriculum_args
        self.dict_ip = dict_ip
        self.think_rate = think_rate
        self.dict_mask = dict_mask

        self.tokenizer = tokenizer

        total_size = 0
        tutorial_size = None
        # print (len(raw_dataset))
        for i in range(len(raw_dataset)):
            assert len(raw_dataset[i][1]) == tutorial_size or tutorial_size is None
            tutorial_size = len(raw_dataset[i][1])
            total_size += tutorial_size
        
        self.tutorial_size = tutorial_size
        self.size = total_size
        self.n_textbook = len(raw_dataset)
        print(f"Total size: {self.size}", flush=True)

    def __len__(self):
        return self.size
    
    
    def get_sample(self, idx):
        assert idx < self.size
        
        if self.data_args.training_type == 'IM' and self.n_textbook > 1:
            tutorial_ind = idx % self.n_textbook
            sample_ind = (idx // self.n_textbook) 
        else:
            tutorial_ind = idx // self.tutorial_size
            sample_ind = idx % self.tutorial_size

        # print (tutorial_ind, sample_ind, idx, self.n_textbook, len(self.raw_dataset[tutorial_ind][1]))
        meta = self.raw_dataset[tutorial_ind][0]
        sample = self.raw_dataset[tutorial_ind][1][sample_ind]
        return meta, sample, tutorial_ind, sample_ind

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.get_item(idx)
        elif isinstance(idx, list):
            return [self.get_item(i) for i in idx]
    
    def __getitems__(self, keys):
        return self.__getitem__(keys)
    
    def get_item(self, idx):
        np.random.seed(idx)

        progress = idx / self.size
        meta, sample, tutorial_ind, sample_ind = self.get_sample(idx)

        format_config = generate_format_config(progress, meta, self.curriculum_args, self.dict_ip, self.think_rate, self.dict_mask)

        q, a = format_tutorial_v5(
            meta=meta,
            sample=sample,
            task_name=f"language_task_{tutorial_ind}",
            data_args=self.data_args,
            curriculum_args=self.curriculum_args,
            format_config=format_config
        )

        q_encoded = self.tokenizer.encode(q)
        a_encoded = self.tokenizer.encode(a)

        return {
            'input_ids': q_encoded,
            'labels': a_encoded
        }

def encode_dataset(
        tokenizer,
        ds_type,
        data_args,
        curriculum_args,
        dict_ip=None,
        think_rate=None,
        dict_mask=None,
    ):

    ds_path = data_args.eval_path if 'test' in ds_type else data_args.train_path
    
    if ds_type == 'train':
        ds_path = data_args.train_path
        n_textbook = data_args.train_n_textbook
        n_sample = data_args.train_n_sample
        ds_source = 'train'

    elif 'train' in ds_type: # generating in-domain eval dataset
        ds_path = data_args.train_path
        n_textbook = data_args.eval_n_textbook
        n_sample = data_args.eval_n_sample
        ds_source = 'train'
    
    elif 'test' in ds_type:
        ds_path = data_args.eval_path
        n_textbook = data_args.eval_n_textbook
        n_sample = data_args.eval_n_sample
        ds_source = 'test'

    else:
        raise NotImplementedError("Invalid dataset type")

    eval_n_textbook = data_args.eval_n_textbook
    eval_n_sample = data_args.eval_n_sample

    assert n_sample % n_textbook == 0
    assert eval_n_sample % eval_n_textbook == 0
    
    print(f"Loading dataset {ds_type} with {n_textbook} textbooks", flush=True)

    # load the dataset
    if n_textbook > 1:  
        if data_args.training_type == 'ICL':
            ds_size = 10 ** int(np.log10(n_textbook - 1) + 1)
            ds_data_path = os.path.join(ds_path, f'{ds_source}_{ds_size}.pkl')
            assert os.path.isfile(ds_data_path), f"Dataset file not found: {ds_data_path}"
            dataset = pickle.load(open(ds_data_path, 'rb'))
        elif data_args.training_type == 'IM':
            ds_size = 10 ** int(np.log10(n_sample - 1) + 1)
            ds_data_path = os.path.join(ds_path, f'{ds_source}_{ds_size}.pkl')
            assert os.path.isfile(ds_data_path), f"Dataset file not found: {ds_data_path}"
            dataset = pickle.load(open(ds_data_path, 'rb'))
    else:
        # Assuming the cutoff is at sample level
        print("Loading dataset with 1 textbook")
        ds_size = 10 ** int(np.log10(n_sample - 1) + 1)
        ds_data_path = os.path.join(ds_path, f'{ds_source}_{ds_size}.pkl')
        assert os.path.isfile(ds_data_path), f"Dataset file not found: {ds_data_path}"
        dataset = pickle.load(open(ds_data_path, 'rb'))

    dataset = dataset[:n_textbook]
    print(n_sample, n_textbook)
    print("Trimming each tutorial to {} samples".format(n_sample // n_textbook), flush=True)
    for i in range(len(dataset)):
        dataset[i][1] = dataset[i][1][:(n_sample // n_textbook)]
    
    print(f"Encoding dataset {ds_type} \
            ({len(dataset)} tutorial, {n_sample} samples),\n \
            CoT on: {data_args.cot_on},\n \
            dict_ip: {dict_ip},\n \
            think_rate: {think_rate},\n \
            dict_mask: {dict_mask},\n \
            think_token_curriculum: {curriculum_args.think_token_curriculum},\n \
            dict_mask_rate_curriculum: {curriculum_args.dictionary_mask_rate_curriculum},\n \
            dict_irrelevant_rate_curriculum: {curriculum_args.dict_irrelevant_rate_curriculum},\n \
            think_token_direction_per_level: {curriculum_args.think_token_direction_per_level},\n \
            think_token_direction: {curriculum_args.think_token_direction},\n \
            dictionary_mask_token_direction: {curriculum_args.dictionary_mask_token_direction},\n \
            dictionary_mask_config: {curriculum_args.dictionary_mask_config},\n \
            ", flush=True)
    
    ret_ds = OnTheFlyDataset(
        dataset,
        data_args,
        curriculum_args,
        tokenizer,
        dict_ip,
        think_rate,
        dict_mask
    )

    return ret_ds, dataset


############## Test func ################
def main():
    return

if __name__ == "__main__":
    main()