import argparse
import os
import re
from tqdm import tqdm
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from comet import download_model, load_from_checkpoint
# from comet.cli import mbr
from utils import get_prompt


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-dn', '--dataset-name', type=str, default='mt-align-study-w-idiom-1203')
    parser.add_argument('-dp', '--dataset-path', type=str, default='data/raw/{DATASET_NAME}')
    parser.add_argument('-s', '--split', type=str, default='train')
    parser.add_argument('-mn', '--model-name', type=str, default='TowerInstruct-7B-v0.2')
    parser.add_argument('-hp', '--hypotheses-path', type=str, default='data/training/{DATASET_NAME}/{MODEL_NAME}')
    parser.add_argument('-tf', '--translations-file', type=str, default='hypotheses_greedy.txt')
    parser.add_argument('-cf', '--candidates-file', type=str, default='hypotheses_N20_t0.9_p0.6.txt')
    parser.add_argument('-scp', '--scorer-path', type=str, default='Unbabel/wmt22-comet-da')
    parser.add_argument('-m', '--method', type=str, default='reference-based')
    parser.add_argument('-msd', '--min-score-diff', type=float, default=0.01)
    args = parser.parse_args()

    # Retrieve generation parameters from file name
    num_candidates = int(re.search(r'_N(\d+)', args.candidates_file).group(1))
    temperature = re.search(r't(\d+\.\d+)', args.candidates_file).group(1)
    top_p = re.search(r'p(\d+\.\d+)', args.candidates_file).group(1)

    # Load data
    print('\n==========> Loading data...')
    dataset_path = args.dataset_path.format(DATASET_NAME=args.dataset_name)
    dataset = load_dataset(dataset_path)
    dataset = dataset[args.split]
    sources = dataset['src']
    try:
        references = dataset['tgt']
    except:
        references = dataset['ref']
    hyp_path = args.hypotheses_path.format(
        DATASET_NAME=args.dataset_name, 
        MODEL_NAME=args.model_name
    )
    with open(f'{hyp_path}/{args.translations_file}', 'r') as f:
        translations = [tsl.replace('\\n', '\n') for tsl in f.read().split('\n')]
    with open(f'{hyp_path}/{args.candidates_file}', 'r') as f:
        candidates = [cdt.replace('\\n', '\n') for cdt in f.read().split('\n')]
    print('==========> Done.\n')

    # Score candidates
    if args.method == 'reference-based': # With access to reference translations        
        # Format data for scoring
        sources_rep = np.repeat(sources, num_candidates)
        references_rep = np.repeat(references, num_candidates)
        scoring_data_cdt = [{'src': src, 'mt': cdt, 'ref': ref} for src, cdt, ref in zip(sources_rep, candidates, references_rep)]
        scoring_data_tsl = [{'src': src, 'mt': cdt, 'ref': ref} for src, cdt, ref in zip(sources, translations, references)]
        
        # Score hypotheses
        print('==========> Scoring hypotheses...')
        model = load_from_checkpoint(download_model(args.scorer_path))
        scores_cdt = np.array(model.predict(scoring_data_cdt, gpus=1).scores).reshape(-1, num_candidates)
        scores_tsl = model.predict(scoring_data_tsl, gpus=1).scores
        print('==========> Done.\n')
        
        # Select best candidates
        candidates_matrix = np.array(candidates).reshape(-1, num_candidates)
        scores_cdt_argmax = scores_cdt.argmax(1)
        best_candidates = candidates_matrix[np.arange(candidates_matrix.shape[0]), scores_cdt_argmax]

        # Build preference dataset
        print('==========> Building preference data...')
        preference_data = {
            'prompt': [],
            'chosen': [],
            'rejected': [],
            'score_chosen': [],
            'score_rejected': []
        }
        for i in tqdm(range(len(dataset))):
            if scores_cdt[i, scores_cdt_argmax[i]] >= scores_tsl[i] + args.min_score_diff:
                preference_data['prompt'].append(get_prompt(sources[i], dataset['lp'][i][:2], dataset['lp'][i][-2:]))
                preference_data['chosen'].append(best_candidates[i] + '<|im_end|>')
                preference_data['rejected'].append(translations[i] + '<|im_end|>')
                preference_data['score_chosen'].append(scores_cdt[i, scores_cdt_argmax[i]])
                preference_data['score_rejected'].append(scores_tsl[i])
        print('==========> Done.\n')

    elif args.method == 'mbr': # Without access to reference translations
        raise NotImplementedError

    # Save preference data
    print('==========> Saving preference dataset...')
    preference_data = Dataset.from_dict(preference_data)
    preference_data = DatasetDict({'train': preference_data})
    preference_data.save_to_disk(f'{hyp_path}/preference-data-N{num_candidates}-t{temperature}-p{top_p}')
    if os.path.exists(f'{hyp_path}/preference-data-N{num_candidates}-t{temperature}-p{top_p}/state.json'):
        os.remove(f'{hyp_path}/preference-data-N{num_candidates}-t{temperature}-p{top_p}/state.json')
    print('==========> Done.\n')
  
  
if __name__ == '__main__':
    main()