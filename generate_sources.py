import argparse
import os
from tqdm import tqdm
from datasets import load_dataset


def get_prompt(
        src, src_lang, tgt_lang, 
        template='<|im_start|>user\\nTranslate the following [SRC_LANG] source text to [TGT_LANG]:\\n' + \
            '[SRC_LANG]: [SRC_SENTENCE]\\n[TGT_LANG]: <|im_end|>\\n<|im_start|>assistant\\n'
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
    print('==========> Loading dataset...')
    dataset = load_dataset(args.dataset_path)
    sources = dataset[args.split]['src']
    src_langs = dataset[args.split]['src_lang']
    tgt_langs = dataset[args.split]['tgt_lang']
    print('==========> Done.\n')

    # Generate source file
    print('==========> Generating source file...')
    instructions_rep = []
    
    for src, src_lang, tgt_lang in tqdm(list(zip(sources, src_langs, tgt_langs))):
        instructions_rep += [get_prompt(src, src_lang, tgt_lang)] * args.num_candidates

    instructions_rep_txt = '\n'.join(instructions_rep)
    sources_txt = '\n'.join(sources)
    print('==========> Done.\n')

    # Save source files
    print('==========> Saving source files...')
    if not os.path.exists('data/'):
        os.makedirs('data/')    
    
    with open('data/sources.txt', 'w') as f:
        f.write(sources_txt)     
    
    with open(
        f'data/repeated_sources_N{args.num_candidates}.txt', 'w') as f:
        f.write(instructions_rep_txt)
    
    print('==========> Done.')


if __name__ == '__main__':
    main()
