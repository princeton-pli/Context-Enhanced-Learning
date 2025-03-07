from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, Optional, Union, List
from transformers import TrainingArguments as TA

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to the teacher model"}
    )
    cache_dir: Optional[str] = field(
        metadata={"help": "cache dir"}
    )
    pad_token_id: Optional[int] = field(
        default=-1,
        metadata={"help": "Size of vocabulary"}
    )
    use_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use lora"}
    )


@dataclass
class TrainingArguments(TA):
    """
    Arguments pertaining to training arguments for distillation
    """

    seed: Optional[int] = field(
        default=42,
        metadata={"help": "Seed to use for training (including data shuffling)"}
    )

    project_name: Optional[str] = field(
        default='incontext_training',
        metadata={"help": "Name of the W&B project"}
    )

    report_to: Optional[str] = field(
        default='wandb',
        metadata={"help": "Whether to log to W&B"}
    )

    doc_mask: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to apply masking on document"}
    )

    eval_strategy: Optional[str] = field(
        default='steps',
        metadata={"help": "Mode of evaluation strategy (no, steps, epoch)"}
    )

    eval_steps: Optional[int] = field(
        default=4,
        metadata={"help": "Frequence of evaluation"}
    )

    eval_on_start: Optional[bool] = field(
        default=True,
        metadata={"help": "Eval on start of training"}
    )

    eval_generation: Optional[bool] = field(
        default=False,
        metadata={"help": "Eval on generation"}
    )

    disable_tqdm: Optional[bool] = field(
        default=True,
        metadata={"help": "Disable TQDM for cleaner logging"}
    )

    logging_first_step: Optional[bool] = field(
        default=True,
        metadata={"help": "Log metrics at the first step"}
    )

    logging_steps: Optional[int] = field(
        default=5,
        metadata={"help": "Log metrics every x steps"}
    )

    include_inputs_for_metrics: Optional[bool] = field(
        default=True,
        metadata={"help": "Include inputs in the metrics computation, must be true for evaluation on generation"}
    )

    shuffle_train: Optional[bool] = field(
        default=False,
        metadata={"help": "Shuffle training data"}
    )

    save_only_model: Optional[bool] = field(
        default=True,
        metadata={"help": "Save only model"}
    )


@dataclass
class DataArguments:
    #dataset to train on
    train_path: Optional[str] = field(
        default='',
        metadata={"help": " Path to the dataset"}
    )
    
    eval_path: Optional[str] = field(
        default='',
        metadata={"help": " Path to the dataset"}
    )

    textbook_mode: Optional[str] = field(
        default='full',
        metadata={"help": " Mode of textbook generation (full / activated)"}
    )

    cot_on: Optional[bool] = field(
        default=False,
        metadata={"help": " Whether to introduce cot in training"}
    )
    
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    max_sequence_length: Optional[int] = field(
        default=1024,
        metadata={"help": "Context length"}
    )

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    train_n_textbook: Optional[int] = field(
        default=1000,
        metadata={"help": "Number of textbooks in training"}
    )

    train_n_sample: Optional[int] = field(
        default=10000,
        metadata={"help": "Number of samples in training"}
    )

    eval_n_textbook: Optional[int] = field(
        default=1000,
        metadata={"help": "Number of textbooks in testing"}
    )

    dict_permutation: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to permute the dictionaries"}
    )

    chars_permutation: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to permute the dictionaries"}
    )

    eval_n_sample: Optional[int] = field(
        default=10000,
        metadata={"help": "Number of samples in testing"}
    )

    abs_letters: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use abstract letters"}
    )

    training_type: Optional[str] = field(
        default='ICL',
        metadata={"help": "ICL/IM"}
    )

    

@dataclass
class CurriculumArguments:

    dict_irrelevant_rate_curriculum: Optional[str] = field(
        default='on',
        metadata={"help": "Curriculum of irrelevant dictionaries, choose from 'linear', 'exp', 'log', 'sigmoid'"}
    )
    
    think_token: Optional[str] = field(
        default='<|reserved_special_token_100|>',
        metadata={"help": "Token to represent think"}
    )

    think_token_curriculum: Optional[str] = field(
        default='on',
        metadata={"help": "Curriculum of introducing think tokens, choose from 'linear', 'exp', 'log', 'sigmoid'"}
    )
    
    think_token_direction: Optional[str] = field(
        default='forward',
        metadata={"help": "Direction of introducing thinking tokens, choose from 'forward', 'backward', 'even'"}
    )

    think_token_direction_per_level: Optional[str] = field(
        default='forward',
        metadata={"help": "Direction of introducing thinking tokens per level, choose from 'forward', 'backward', 'even'"}
    )
        
    cot_pad_token: Optional[str] = field(
        default='*',
        metadata={"help": "Token to represent think"}
    )

    a_cot_max_length: Optional[int] = field(
        default=40,
        metadata={"help": "Max length of a_cot"}
    )

    # For dictionary masking tokens
    dictionary_mask_token_template: Optional[str] = field(
        default='<|reserved_special_token_{index}|>',
        metadata={"help": "Token to represent dictionary mask"}
    )

    dictionary_mask_token_base_index: Optional[int] = field(
        default=101,
        metadata={"help": "Base index for dictionary mask tokens"}
    )

    dictionary_mask_rate_curriculum: Optional[str] = field(
        default='off',
        metadata={"help": "Curriculum of introducing masking tokens for dictionary, choose from 'linear', 'exp', 'log', 'sigmoid'"}
    )

    dictionary_mask_token_direction: Optional[str] = field(
        default='forward',
        metadata={"help": "Direction of introducing thinking tokens, choose from 'forward', 'backward', 'even'"}
    )

    dictionary_mask_config: Optional[str] = field(
        default='0-0-0-0-0',
        metadata={"help": "Rate of introducing masking tokens for dictionary, seperated by -"}
    )

    dictionary_mask_indices:  Optional[str] = field(
        default='same',
        metadata={"help": "Whether to use different mask indices for different dictionaries: same/different/indices separated by hyphens"}
    )

@dataclass
class Unionable:
    def __or__(self, other):
        return self.__class__(**asdict(self) | asdict(other))