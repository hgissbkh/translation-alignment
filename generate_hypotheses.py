import argparse
import os
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams
from utils import get_prompt


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-dn', '--dataset-name', type=str, default='mt-align-study-w-idiom-1203')
    parser.add_argument('-dp', '--dataset-path', type=str, default='data/raw/{DATASET_NAME}')
    parser.add_argument('-s', '--split', type=str, default='train')
    parser.add_argument('-c', '--checkpoint', type=str, default='unaligned')
    parser.add_argument('-hp', '--hyperparameters', type=str, default='bs32_lr1e-05_wu0.0')
    parser.add_argument('-mn', '--model-name', type=str, default='TowerInstruct-7B-v0.2')
    parser.add_argument('-mp', '--model-path', type=str, default='models/{MODEL_NAME}/{HYPERPARAMETERS}/{CHECKPOINT}')
    parser.add_argument('-N', '--num-candidates', type=int, default=1)
    parser.add_argument('-t', '--temperature', type=float, default=0)
    parser.add_argument('-p', '--top-p', type=float, default=1)
    parser.add_argument('-st', '--stage', type=str, default='training')
    parser.add_argument('-sp', '--save-path', type=str, default='data/{STAGE}/{DATASET_NAME}/{MODEL_NAME}/{HYPERPARAMETERS}/{CHECKPOINT}')
    args = parser.parse_args()

    # Load data
    print('\n==========> Loading dataset...')
    dataset_path = args.dataset_path.format(DATASET_NAME=args.dataset_name)
    dataset = load_dataset(dataset_path)
    dataset = dataset[args.split]
    print('==========> Done.\n')

    # Create prompts
    print('==========> Generating promtps...')
    src_langs = [lp[:2] for lp in dataset['lp']]
    tgt_langs = [lp[-2:] for lp in dataset['lp']]
    prompts = []
    if args.model_name.startswith('ALMA'):
        template_type = 'ALMA'
    else:
        template_type = 'Tower'
    for src, src_lang, tgt_lang in tqdm(list(zip(dataset['src'], src_langs, tgt_langs))):
        prompts += [get_prompt(src, src_lang, tgt_lang, template_type)] * args.num_candidates
    print('==========> Done.\n')
    
    # Load model
    print('==========> Loading model...')
    if (args.checkpoint == 'unaligned') or (args.hyperparameters == 'None'):
        model_path = args.model_path.format(
            MODEL_NAME=args.model_name,
            HYPERPARAMETERS='',
            CHECKPOINT=args.checkpoint
        ).replace('//', '/')
    else:        
        if args.checkpoint == 'aligned':
            model_path = args.model_path.format(
                MODEL_NAME=args.model_name, 
                HYPERPARAMETERS=args.hyperparameters,
                CHECKPOINT=''
            )[:-1]
        elif args.checkpoint.startswith('checkpoint'):
            model_path = args.model_path.format(
                MODEL_NAME=args.model_name, 
                HYPERPARAMETERS=args.hyperparameters,
                CHECKPOINT=args.checkpoint
            )
    llm = LLM(model=model_path, max_model_len=1024, tensor_parallel_size=1)
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
    if args.stage == 'training':
        save_path = args.save_path.format(
            STAGE=args.stage, 
            DATASET_NAME=args.dataset_name,
            MODEL_NAME=args.model_name,
            CHECKPOINT=''
        )
    elif args.stage == 'evaluation':
        if (args.checkpoint == 'unaligned') or (args.hyperparameters == 'None'):
            save_path = args.save_path.format(
                STAGE=args.stage, 
                DATASET_NAME=args.dataset_name,
                MODEL_NAME=args.model_name,
                HYPERPARAMETERS='', 
                CHECKPOINT=args.checkpoint
            ).replace('//', '/') 
        else:
            save_path = args.save_path.format(
                STAGE=args.stage, 
                DATASET_NAME=args.dataset_name,
                MODEL_NAME=args.model_name, 
                HYPERPARAMETERS=args.hyperparameters, 
                CHECKPOINT=args.checkpoint
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
