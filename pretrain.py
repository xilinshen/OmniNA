import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import datasets

import torch
import transformers
from transformers import Trainer, DataCollatorForLanguageModeling
import os

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "### Instruction: {instruction} ### Input: {input} ### Response:"
    ),
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default='XLS/OmniNA-220m')


@dataclass
class DataArguments:
    train_data_path: str = field(default=None, metadata={"help": "Path to the training dataset."})    
    val_data_path: str = field(default=None, metadata={"help": "Path to the validation dataset."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def _tokenize_fn():
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(x): 
    instruction = "Annotate the following sequence."
    prompt_input = PROMPT_DICT['prompt_input']
    sources = [prompt_input.format_map({"instruction":instruction, "input":input}) for input in x['sequence']]
    targets = [f"{output}{DEFAULT_EOS_TOKEN}" for output in x['response']]

    examples = [s + t for s, t in zip(sources, targets)]
    return dict(examples = examples)

def tokenize_data(x):
    examples_tokenized = [tokenizer(strings, truncation=True, padding='longest', max_length=tokenizer.model_max_length) for strings in x['examples']]
    input_ids = [i['input_ids'] + [tokenizer.eos_token_id] for i in examples_tokenized]
    attention_mask = [i['attention_mask'] + [1] for i in examples_tokenized]
    return {"input_ids": input_ids, 'attention_mask':attention_mask}

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for pretraing."""
    train_dataset = datasets.load_dataset("csv", data_files=data_args.train_data_path)
    train_dataset = train_dataset['train']
    train_dataset = train_dataset.map(
            preprocess,
            batched=True,
            num_proc=96,
            remove_columns=train_dataset.features.keys(),
            load_from_cache_file=True)
    train_dataset = train_dataset.map(
            tokenize_data,
            batched=True,
            num_proc=96,
            remove_columns=train_dataset.features.keys(),
            load_from_cache_file=True)
    
    val_dataset = datasets.load_dataset("csv", data_files=data_args.val_data_path)
    val_dataset = val_dataset['train']
    val_dataset = val_dataset.map(
            preprocess,
            batched=True,
            num_proc=96,
            remove_columns=val_dataset.features.keys(),
            load_from_cache_file=True)
    val_dataset = val_dataset.map(
            tokenize_data,
            batched=True,
            num_proc=96,
            remove_columns=val_dataset.features.keys(),
            load_from_cache_file=True)
    
    #tokenized_dataset = datasets.load_from_disk(data_args.data_path)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, mlm_probability=0.0)
    return dict(train_dataset=train_dataset, 
                eval_dataset=val_dataset, 
                data_collator=data_collator)

def tokenize_data(x):
    examples_tokenized = [tokenizer(strings, truncation=True, padding='longest', max_length=tokenizer.model_max_length) for strings in x['examples']]
    input_ids = [i['input_ids'] for i in examples_tokenized]
    attention_mask = [i['attention_mask'] for i in examples_tokenized]
    return {"input_ids": input_ids, 'attention_mask':attention_mask}

def train():
    global tokenizer

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()