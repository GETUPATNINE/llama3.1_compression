import random
import numpy as np
import torch
import json
import glob
import os

from datasets import load_dataset
from torch.utils.data.dataset import Dataset

def get_c4(tokenizer, n_samples, seq_len):
    # traindata = load_dataset(
    #     'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    # )
    traindata = load_dataset('data/c4', split='train')
    
    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(traindata[i]['text'], return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len )
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])
    return torch.cat(tokenized_samples, dim=0)

def get_bookcorpus(tokenizer, n_samples, seq_len):
    traindata = load_dataset(
        'bookcorpus', split='train'
    )
    
    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(traindata[i]['text'], return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])
    return torch.cat(tokenized_samples, dim=0 )

def get_diabetes_dataset(data_dir='data/diabetes/with_info/', pattern='diabetes_*.json'):
    """Load all diabetes dataset files matching the pattern."""
    all_data = []
    file_paths = glob.glob(os.path.join(data_dir, pattern))
    
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            data = json.load(f)
            all_data.extend(data)
    print(len(all_data))

    return all_data

def get_diabetes(tokenizer, n_samples, seq_len):
    """Process diabetes dataset for LLM pruning."""
    all_data = get_diabetes_dataset()
    
    if len(all_data) < n_samples:
        indices = np.random.choice(len(all_data), n_samples, replace=True)
    else:
        indices = np.random.choice(len(all_data), n_samples, replace=False)
    
    tokenized_samples = []
    
    for idx in indices:
        sample = all_data[idx]
        full_text = f"""Below is an instruction that describes a task, along with input data. Write a response that appropriately completes the request.

### Instruction:
{sample['instruction']}

### Input:
{sample['input']}

### Response:
{sample['output']}"""
        
        tokenized_sample = tokenizer(full_text, return_tensors='pt')
        
        if tokenized_sample.input_ids.shape[1] < seq_len:
            pad_length = seq_len - tokenized_sample.input_ids.shape[1]
            padding = torch.full((1, pad_length), tokenizer.pad_token_id, dtype=torch.long)
            padded_sample = torch.cat([tokenized_sample.input_ids, padding], dim=1)
            tokenized_samples.append(padded_sample[:, :seq_len])
        elif tokenized_sample.input_ids.shape[1] > seq_len:
            tokenized_samples.append(tokenized_sample.input_ids[:, :seq_len])
        else:
            tokenized_samples.append(tokenized_sample.input_ids)
    
    return torch.cat(tokenized_samples, dim=0)

def get_examples(dataset, tokenizer, n_samples, seq_len=128):
    if dataset == 'c4':
        return get_c4(tokenizer, n_samples, seq_len)
    elif dataset == 'bookcorpus':
        return get_bookcorpus(tokenizer, n_samples, seq_len)
    elif dataset == 'diabetes':
        return get_diabetes(tokenizer, n_samples, seq_len)
    else:
        raise NotImplementedError