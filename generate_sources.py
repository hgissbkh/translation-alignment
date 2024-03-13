import argparse
import os
from tqdm import tqdm
import numpy as np
from vllm import LLM
from datasets import load_dataset


def get_instruction(
        src, src_lang, tgt_lang, 
        template='<|im_start|>user\\nTranslate the following [SRC_LANG] source text to [TGT_LANG]:\\n[SRC_LANG]: [SRC_SENTENCE]\\n[TGT_LANG]: <|im_end|>\n'
):
    lang_dict = {
        'de': 'German',
        'en': 'English',
        'es': 'Spanish',
        'fr': 'French',
        'it': 'Italian',
        'pt': 'Portuguese'
    }
    instruction = template.replace(
        '[SRC_SENTENCE]', src
    ).replace(
        '[SRC_LANG]', lang_dict[src_lang]
    ).replace(
        '[TGT_LANG]', lang_dict[tgt_lang]
    )
    return instruction
    

def main():
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset-path', type=str, default='sardinelab/mt-align-study-w-idiom-1203')
    parser.add_argument('-s', '--split', type=str, default='train')
    parser.add_argument('-N', '--num-candidates', type=int, default=20)
    args = parser.parse_args()

    # Load data
    print('Loading dataset...')
    dataset = load_dataset(args.dataset_path)
    sources = dataset[args.split]['src']
    src_langs = dataset[args.split]['src_lang']
    tgt_langs = dataset[args.split]['tgt_lang']
    print('Done.\n')

    # Generate source file
    print('Generating source file...')
    repeated_instructions_txt = ''
    for src, src_lang, tgt_lang in tqdm(list(zip(sources, src_langs, tgt_langs))):
        repeated_instructions_txt += get_instruction(src, src_lang, tgt_lang) * args.num_candidates
    print('Done.\n')

    # Save source file
    print('Saving source file...')
    with open(
        'data/' + args.dataset_path.split('/')[1].replace('-', '_') 
        + '_' + args.dataset_path.split('/')[1].replace('-', '_') 
        + '_' + str(args.num_candidates) + '_sources.txt', 'w'
    ) as f:
        f.write(repeated_instructions_txt)
    print('Done.')


if __name__ == '__main__':
    main()
