import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import datasets
import os

import torch
import transformers
from transformers import Trainer

IGNORE_INDEX = -100

PROMPT_DICT = {
    "prompt_input": (
       "### Instruction: {instruction} ### Input: {input} ### Question: {question} ### Response:"
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

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [tokenizer(text, truncation=True, padding='longest', max_length=tokenizer.model_max_length, return_tensors = 'pt') for text in strings]
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
    """Preprocess the data by tokenizing."""
    instruction = "Annotate the following sequence"
    prompt_input = PROMPT_DICT['prompt_input']
    sources = [prompt_input.format_map({"instruction":instruction, "input":input, "question":question}) for question,input in zip(x['question'], x['sequence'])]
    targets = [f"{output} {tokenizer.eos_token}" for output in x['response']]

    examples = [s + t for s, t in zip(sources, targets)]

    # mask out the sequence and train only the response. 
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX        
    return dict(input_ids=input_ids, labels=labels)

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([torch.tensor(instance[key]) for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
            
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = datasets.load_dataset("csv", data_files=data_args.train_data_path)
    train_dataset = train_dataset['train']
    train_dataset = train_dataset.map(
            preprocess,
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
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    return dict(train_dataset=train_dataset, 
                eval_dataset=val_dataset, 
                data_collator=data_collator)

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
    
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()