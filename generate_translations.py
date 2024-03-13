import argparse
import os
from tqdm import tqdm
import numpy as np
from vllm import LLM, SamplingParams
from datasets import load_dataset


def main():
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--source-file', type=str, default='data/mt_align_study_w_idiom_1203_mt_align_study_w_idiom_1203_20_sources.txt')
    parser.add_argument('-m', '--model-path', type=str, default='Unbabel/TowerInstruct-7B-v0.2')
    parser.add_argument('-t', '--temperature', type=float, default=0.9)
    parser.add_argument('-p', '--top-p', type=float, default=0.6)
    parser.add_argument('-mt', '--max-tokens', type=float, default=1024)
    parser.add_argument('-g', '--gpu', type=str, default='N/A')
    args = parser.parse_args()

    # Set up GPU
    if args.gpu != 'N/A':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Load model
    print('Loading model...')
    llm = LLM(model=args.model_path, tensor_parallel_size=1)
    print('Done.\n')

    # Load source data
    print('Loading source data...')
    with open(args.source_file, 'r') as f:
        prompts = [line.strip().replace('\\n', '\n') for line in tqdm(f)]
    print('Done.\n')

    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens
    )
    
    # Generate candidates
    print('Generating candidates...')
    outputs = llm.generate(prompts, sampling_params)
    generations_txt = '\n'.join([o.outputs[0].text.strip() for o in outputs])
    print('Done.\n')

    # Save candidates file
    print('Saving candidates file...')
    with open(args.source_file.replace('sources', 'candidates'), 'w') as f:
        f.write(generations_txt)
    print('Done.')

if __name__ == '__main__':
    main()
