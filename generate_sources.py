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
    parser.add_argument('-sd', '--save-directory', type=str, default='data/train')
    parser.add_argument('-gr', '--generate-refs', action='store_true')
    args = parser.parse_args()

    # Load data
    print('==========> Loading dataset...')
    dataset = load_dataset(args.dataset_path)
    dataset = dataset[args.split]
    print('==========> Done.\n')

    # Generate source file
    print('==========> Generating .txt files...')
    sources = dataset['src']

    try:
        references = dataset['tgt']
    except:
        references = dataset['ref']

    if 'prompt' in dataset.features:
        prompts = dataset['prompt']
        prompts_rep = [prompt for prompt in prompts for _ in repeat(None, args.num_candidates)]
    else:    
        src_langs = dataset['src_lang']
        tgt_langs = dataset['tgt_lang']
        prompts_rep = [] 
        
        for src, src_lang, tgt_lang in tqdm(list(zip(sources, src_langs, tgt_langs))):
            prompts_rep += [get_prompt(src, src_lang, tgt_lang).replace('\n', '\\n')] * args.num_candidates

    prompts_rep_txt = '\n'.join(prompts_rep)
    sources_txt = '\n'.join(sources)
    references_txt = '\n'.join(references)
    print('==========> Done.\n')

    # Save source files
    print('==========> Saving .txt files...')
    with open(f'{args.save_directory}/sources.txt', 'w') as f:
        f.write(sources_txt)     
    
    with open(
        f'{args.save_directory}/repeated_sources_N{args.num_candidates}.txt', 'w') as f:
        f.write(prompts_rep_txt)
    
    with open(f'{args.save_directory}/references.txt', 'w') as f:
        f.write(references_txt)     

    print('==========> Done.')


if __name__ == '__main__':
    main()
