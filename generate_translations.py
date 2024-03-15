import argparse
import os
from tqdm import tqdm
import numpy as np
from vllm import LLM, SamplingParams


def main():
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--source-file', type=str, default='data/mt_align_study_w_idiom_1203_mt_align_study_w_idiom_1203_20_sources_rep.txt')
    parser.add_argument('-m', '--model-path', type=str, default='Unbabel/TowerInstruct-7B-v0.2')
    parser.add_argument('-ml', '--model-path-local', type=str, default='/linkhome/rech/genrce01/urb81vh/.cache/huggingface/hub/[MODEL]/snapshots/6ea1d9e6cb5e8badde77ef33d61346a31e6ff1d4')
    parser.add_argument('-t', '--temperature', type=float, default=0.9)
    parser.add_argument('-p', '--top-p', type=float, default=0.6)
    parser.add_argument('-mt', '--max-tokens', type=float, default=1024)
    parser.add_argument('-g', '--gpu', type=str, default='N/A')
    args = parser.parse_args()

    # Set up GPU
    if args.gpu != 'N/A':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Load model
    print('==========> Loading model...')
    model_path_local = args.model_path_local.replace('[MODEL]', 'models--' + args.model_path.replace('/', '--'))
    llm = LLM(model=model_path_local, max_model_len=1024, tensor_parallel_size=1)        
    print('==========> Done.\n')

    # Load source data
    print('==========> Loading source data...')
    with open(args.source_file, 'r') as f:
        prompts = [line.strip().replace('\\n', '\n') for line in f]
    print('==========> Done.\n')

    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens
    )
    
    # Generate candidates
    print('==========> Generating candidates...')
    outputs = llm.generate(prompts, sampling_params)
    generations = [o.outputs[0].text.strip() for o in outputs]
    generations_txt = '\n'.join(generations)
    print('==========> Done.\n')

    # Save candidates file
    print('==========> Saving candidates file...')
    with open(
        args.source_file.replace('sources', '') + str(args.temperature) 
        + '_' + str(args.top_p) + '_candidates', 'w'
    ) as f:
        f.write(generations_txt)
    print('==========> Done.')

if __name__ == '__main__':
    main()
