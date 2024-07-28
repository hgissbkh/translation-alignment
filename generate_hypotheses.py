import argparse
import os
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams
from utils import get_prompt


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets-cache-path', type=str, default='~/.cache/huggingface/datasets')
    parser.add_argument('--hf-dataset-path', type=str, default='hgissbkh/WMT22-23-Test-v2')
    parser.add_argument('--language-pairs', type=str, default='cs-en,de-en,is-en,ru-en,zh-en')
    parser.add_argument('--models-cache-path', type=str, default='~/.cache/huggingface/hub')
    parser.add_argument('--hf-model-path', type=str, default='hgissbkh/ALMA-13B-SFT-HW')
    parser.add_argument('-N', '--num-candidates', type=int, default=1)
    parser.add_argument('-t', '--temperature', type=float, default=0)
    parser.add_argument('-p', '--top-p', type=float, default=1)
    parser.add_argument('--save-path', type=str, default='data/{STAGE}/{DATASET_NAME}/{MODEL_NAME}')
    parser.add_argument('--stage', type=str, default='evaluation')
    args = parser.parse_args()

    # Load data
    print('\n==========> Loading dataset...')
    if args.stage == 'evaluation':
        split = 'test'
    else:
        split = 'train'
    dataset = load_dataset(args.hf_dataset_path)[split]
    print('==========> Done.\n')

    # Create prompts
    print('==========> Generating promtps...') 
    src_langs = [lp[:2] for lp in dataset['lp']]
    tgt_langs = [lp[-2:] for lp in dataset['lp']]
    sources = dataset['src']
    prompts = []
    for src, src_lang, tgt_lang in tqdm(list(zip(sources, src_langs, tgt_langs))):
        prompts += [get_prompt(src, src_lang, tgt_lang, 'ALMA')] * args.num_candidates
    print('==========> Done.\n')
    
    # Load model
    print('==========> Loading model...')
    llm = LLM(model=args.hf_model_path, max_model_len=1024, tensor_parallel_size=1)
    print('==========> Done.\n')

    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=1024
    )

    # Generate candidates
    print('==========> Generating hypotheses...')    
    outputs = llm.generate(prompts, sampling_params)
    generations = [o.outputs[0].text.strip() for o in outputs]
    print('==========> Done.\n')
        
    # Save hypotheses as .txt
    print('==========> Saving hypotheses...')
    save_path = args.save_path.format(
        STAGE=args.stage,
        DATASET_NAME=args.hf_dataset_path.split("/")[1],
        MODEL_NAME=args.hf_model_path.split("/")[1]
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if args.temperature == 0:
        file_name = 'hypotheses_greedy.txt'
    else:
        file_name = f'hypotheses_N{args.num_candidates}_t{args.temperature}_p{args.top_p}.txt'
    with open(f'{save_path}/{file_name}', 'w') as f:
        generations = [gen.replace('\n', '\\n') for gen in generations]
        f.write('\n'.join(generations))
    print('==========> Done.\n')


if __name__ == '__main__':
    main()
