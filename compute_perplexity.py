import argparse
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

from utils import get_prompt


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--dataset-path', type=str, default='data/raw/wmt22-23-test')
    parser.add_argument('-mp', '--model-path', type=str, default='models/{STATE}')
    parser.add_argument('-ms', '--model-state', type=str, default='aligned')
    parser.add_argument('-tp', '--translations-path', type=str, default='data/evaluation/{STATE}')
    parser.add_argument('-tf', '--translations-file', type=str, default='hypotheses_greedy.txt')
    parser.add_argument('-ts', '--translations-state', type=str, default='aligned')
    # parser.add_argument('-b', '--batch_size', type=int, default=8)
    args = parser.parse_args()

    # Load data
    print('==========> Loading data...')
    dataset = load_dataset(args.dataset_path)['train']
    tsl_path = args.translations_path.format(STATE=args.translations_state)
    with open(f'{tsl_path}/{args.translations_file}', 'r') as f:
        mts = [mt.replace('\\n', '\n') for mt in f.read().split('\n')]
    print('==========> Done.\n')

    # Create prompts
    print('==========> Generating promtps...')
    src_langs = [lp[:2] for lp in dataset['lp']]
    tgt_langs = [lp[-2:] for lp in dataset['lp']]
    prompts = []
    for src, src_lang, tgt_lang in tqdm(list(zip(dataset['src'], src_langs, tgt_langs))):
        prompts.append(get_prompt(src, src_lang, tgt_lang, template_type='ALMA'))
    print('==========> Done.\n')

    # Concatenate prompts and mts
    prompts_mts = [prompt + mt for prompt, mt in zip(prompts, mts)]

    # Load model
    print('==========> Loading model...')
    model_path = args.model_path.format(STATE=args.model_state)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model = model.to(torch.bfloat16)
    model = model.to('cuda')
    model.eval()
    print('==========> Done.\n')

    # Batch inputs
    # prompts_mts_batches = [
    #     prompts_mts[i * args.batch_size : (i+1) * args.batch_size] 
    #     for i in range(np.ceil(len(prompts_mts) / args.batch_size).astype(int))
    # ]

    # Compute perplexities
    perplexities = []
    
    # for prompts_mts_batch in prompts_mts_batches:
    for prompt_mt in tqdm(prompts_mts):
        inputs = tokenizer(prompt_mt, return_tensors='pt')
        input_ids = inputs.input_ids
        input_ids = input_ids.to('cuda')
        
        with torch.no_grad():
            outputs = model(input_ids)

        
        print(dir(outputs))

        print(outputs.last_hidden_state)

        print(outputs.last_hidden_state.shape)


        
        #perplexity = torch.exp(loss)
        #perplexities.append(perplexity)

        #print(perplexity)

        break

if __name__ == '__main__':
    main()
