# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import numpy as np
import random
import torch
import json
from datasets import load_dataset

# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

# Load and process c4 dataset
def get_c4(nsamples, seed, seqlen, tokenizer):
    # Load train and validation datasets
    traindata = load_dataset('allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    valdata = load_dataset('allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Prepare validation dataset
    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc

def get_alzheimer(nsamples, seed, seqlen, tokenizer):
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Load the Alzheimer dataset from jsonl file
    alzheimer_data = []
    with open('data/spam_classification.jsonl', 'r') as f:
        for line in f:
            alzheimer_data.append(json.loads(line))
    
    # Prepare the training samples
    trainloader = []
    
    # Split data for training and testing (80/20 split)
    random.shuffle(alzheimer_data)
    split_idx = int(len(alzheimer_data) * 0.8)
    train_data = alzheimer_data[:split_idx]
    test_data = alzheimer_data[split_idx:]
    
    # Create training samples
    for _ in range(min(nsamples, len(train_data))):
        idx = random.randint(0, len(train_data) - 1)
        sample = train_data[idx]
        
        # Format the text as instruction + input
        if sample.get("input", ""):
            full_text = f"Instruction: {sample['instruction']}\nInput: {sample['input']}\nOutput: {sample['output']}"
        else:
            full_text = f"Instruction: {sample['instruction']}\nOutput: {sample['output']}"
        
        # Tokenize the text
        tokenized = tokenizer(full_text, return_tensors='pt')
        
        # Handle sequences longer than seqlen
        if tokenized.input_ids.shape[1] > seqlen:
            # Take the first seqlen tokens
            inp = tokenized.input_ids[:, :seqlen]
        else:
            # Pad shorter sequences
            padding_length = seqlen - tokenized.input_ids.shape[1]
            inp = torch.nn.functional.pad(
                tokenized.input_ids, 
                (0, padding_length), 
                'constant', 
                tokenizer.pad_token_id
            )
        
        # Create target by copying input but masking all tokens except the last
        tar = inp.clone()
        tar[:, :-1] = -100  # Mask all except last token
        
        trainloader.append((inp, tar))
    
    # Prepare test samples
    test_texts = []
    for sample in test_data:
        if sample.get("input", ""):
            test_texts.append(f"Instruction: {sample['instruction']}\nInput: {sample['input']}\nOutput: {sample['output']}")
        else:
            test_texts.append(f"Instruction: {sample['instruction']}\nOutput: {sample['output']}")
    
    # Tokenize test data
    testenc = tokenizer(' '.join(test_texts), return_tensors='pt')
    
    # Ensure test sequence is a multiple of seqlen for easier processing
    max_test_len = (testenc.input_ids.shape[1] // seqlen) * seqlen
    testenc = testenc.input_ids[:, :max_test_len]
    testenc = TokenizerWrapper(testenc)
    
    return trainloader, testenc

# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)
    if "alzheimer" in name:
        return get_alzheimer(nsamples, seed, seqlen, tokenizer)