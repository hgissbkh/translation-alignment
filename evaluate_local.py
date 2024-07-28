import argparse
import os
import pickle
import torch
from datasets import load_dataset
from comet import download_model, load_from_checkpoint


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf-dataset-path', type=str, default='/linkhome/rech/genrce01/urb81vh/.cache/huggingface/datasets/hgissbkh___wmt22-23-test-v2/default/0.0.0/701aee0c9289b4205fa785b154190e86037fccf7')
    parser.add_argument('--hf-model-path', type=str, default='/linkhome/rech/genrce01/urb81vh/.cache/huggingface/hub/models--hgissbkh--ALMA-13B-SFT-HW-CPO-Multi-chrF/snapshots/58e3dd76e317c661b79a52bcc55a3d6579e16aef')
    parser.add_argument('--stage', type=str, default='evaluation')
    parser.add_argument('--hypotheses-path', type=str, default='data/{STAGE}/{DATASET_NAME}/{MODEL_NAME}')
    parser.add_argument('--hypotheses-file', type=str, default='hypotheses_greedy.txt')
    parser.add_argument('--system', type=str, default=None)
    parser.add_argument('--hf-scorer-path', type=str, default='Unbabel/wmt23-cometkiwi-da-xxl')
    parser.add_argument('--qe', action='store_true')
    args = parser.parse_args()

    # Load data
    print('\n==========> Loading data...')
    if args.stage == 'evaluation':
        split = 'test'
    else:
        split = 'train'
    ds = load_dataset(args.hf_dataset_path)[split]
    sources = ds['src']
    if args.system is not None:
        hypotheses = ds[args.system]
        num_candidates = 1
    else:        
        if args.hypotheses_file == 'hypotheses_greedy.txt':
            num_candidates = 1
        else:
            num_candidates = int(args.hypotheses_file.split('_')[1][1:])    
        hypotheses_path = args.hypotheses_path.format(
            STAGE=args.stage,
            DATASET_NAME="WMT22-23-Test-v2",
            MODEL_NAME=args.hf_model_path.split('/')[-3].split('--')[-1]
        )
        with open(f'{hypotheses_path}/{args.hypotheses_file}', 'r') as f:
            hypotheses = [tsl.replace('\\n', '\n') for tsl in f.read().split('\n')]
    sources_rep = [src for src in sources for _ in range(num_candidates)]
    print('==========> Done.\n')

    # Score translations with respect to references
    # Load scorer
    print('==========> Loading scorer...')
    model = load_from_checkpoint(download_model(args.hf_scorer_path))
    if args.hf_scorer_path != 'Unbabel/wmt22-comet-da':
        model = model.to(torch.bfloat16)
    print('==========> Done.\n')
    
    # Score hypotheses
    print('==========> Scoring hypotheses...')    
    if args.qe:
        scoring_data = [{'src': src, 'mt': mt} for src, mt in zip(sources_rep, hypotheses)]
    else:
        references_rep = [ref for ref in ds['ref'] for _ in range(num_candidates)]
        scoring_data = [{'src': src, 'mt': mt, 'ref': ref} for src, mt, ref in zip(sources_rep, hypotheses, references_rep)]    
    model_output = model.predict(scoring_data, gpus=1)
    scores = model_output.scores
    print('==========> Done.\n')

    # Save scores
    print('==========> Saving scores...')
    if args.hf_scorer_path == 'Unbabel/wmt23-cometkiwi-da-xxl':
        score_name = 'kiwi'
    elif args.hf_scorer_path == 'Unbabel/XCOMET-XXL':
        score_name = 'xcomet'
    elif args.hf_scorer_path == 'Unbabel/wmt22-comet-da':
        score_name = 'comet'
    if args.system is not None:   
        scores_file = f"{score_name}_{args.system}.pkl"
        save_path = args.hypotheses_path.format(
            STAGE=args.stage,
            DATASET_NAME=args.hf_model_path.split('/')[-3].split('--')[-1],
            MODEL_NAME=""
        )[:-1]
    else: 
        scores_file = args.hypotheses_file.replace('hypotheses', score_name).replace('.txt', '.pkl')
        save_path = hypotheses_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(f'{save_path}/{scores_file}', 'wb') as f:
        pickle.dump(scores, f)
    print('==========> Done.\n')


if __name__ == '__main__':
    main()
