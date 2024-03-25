import argparse
import os
from tqdm import tqdm
from itertools import repeat
from datasets import load_dataset
from utils import get_prompt


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset-path', type=str, default='sardinelab/mt-align-study-w-idiom-1203')
    parser.add_argument('-s', '--split', type=str, default='train')
    parser.add_argument('-N', '--num-candidates', type=int, default=20)
    parser.add_argument('-sd', '--save-directory', type=str, default='data/train/')
    args = parser.parse_args()

    # Load data
    print('==========> Loading dataset...')
    dataset = load_dataset(args.dataset_path)
    dataset = dataset[args.split]
    print('==========> Done.\n')

    # Generate source file
    print('==========> Generating source file...')
    
    if 'prompt' in dataset:
        instructions = dataset['prompt']
        instructions_rep = [prompt for prompt in instructions for _ in repeat(None, args.num_candidates)]
        sources = dataset['src']
    else:    
        sources = dataset['src']
        src_langs = dataset['src_lang']
        tgt_langs = dataset['tgt_lang']
        instructions_rep = [] 
        
        for src, src_lang, tgt_lang in tqdm(list(zip(sources, src_langs, tgt_langs))):
            instructions_rep += [get_prompt(src, src_lang, tgt_lang).replace('\n', '\\n')] * args.num_candidates

    instructions_rep_txt = '\n'.join(instructions_rep)
    sources_txt = '\n'.join(sources)
    print('==========> Done.\n')

    # Save source files
    print('==========> Saving source files...')
    with open(f'{args.save_directory}/sources.txt', 'w') as f:
        f.write(sources_txt)     
    
    with open(
        f'{args.save_directory}/repeated_sources_N{args.num_candidates}.txt', 'w') as f:
        f.write(instructions_rep_txt)
    
    print('==========> Done.')


if __name__ == '__main__':
    main()
