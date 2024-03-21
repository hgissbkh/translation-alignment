import argparse
import re
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from utils import get_prompt


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--hf-dataset-path', type=str, default='sardinelab/mt-align-study-w-idiom-1203')
    parser.add_argument('-s', '--split', type=str, default='train')
    parser.add_argument('-cf', '--candidates-file', type=str, default='data/candidates_N20_t0.9_p0.6.txt')
    parser.add_argument('-sf', '--scores-file', type=str, default='data/scores_comet_mbr_N20_t0.9_p0.6.txt')
    parser.add_argument('-ms', '--min-score', type=float, default=0.8)
    args = parser.parse_args()

    # Load relevant data
    num_candidates = int(re.search(r'N(\d+)', args.candidates_file).group(1))
    dataset = load_dataset(args.hf_dataset_path)
    dataset_train = dataset[args.split]

    with open(args.candidates_file, 'r') as f:
        candidates = f.read().split('\n')
        candidates = [candidates[i:(i+num_candidates)] for i in range(0, len(candidates), num_candidates)]

    with open(args.scores_file, 'r') as f:
        scores = [[float(score) for score in scores.split('\t')] for scores in f.read().split('\n')]

    # Build preference data
    pref_data_train = {
        'prompt': [], 'chosen': [], 'rejected': [], 'score_chosen': [], 'score_rejected': [], 
        'candidates': [], 'scores': [], 'src': [], 'src_lang': [], 'tgt_lang': []
    }

    for cd, sc, src, src_lang, tgt_lang in zip(candidates, scores, dataset_train['src'], dataset_train['src_lang'], dataset_train['tgt_lang']):    
        cd_filt = [s for s, x in zip(cd, sc) if x >= args.min_score] # Only candidates that have a sufficiently high score
        sc_filt = [x for x in sc if x >= args.min_score]
        
        if len(cd_filt) > 0:    
            chosen = cd_filt[np.argmax(sc_filt)]
            rejected = cd[np.argmin(sc)]
            
            if chosen != rejected:
                pref_data_train['prompt'].append(get_prompt(src, src_lang, tgt_lang))
                pref_data_train['chosen'].append(chosen)
                pref_data_train['rejected'].append(rejected)
                pref_data_train['score_chosen'].append(max(sc_filt))
                pref_data_train['score_rejected'].append(min(sc_filt))
                pref_data_train['candidates'].append(cd)
                pref_data_train['scores'].append(sc)
                pref_data_train['src'].append(src)
                pref_data_train['src_lang'].append(src_lang)
                pref_data_train['tgt_lang'].append(tgt_lang)
    
    pref_data_train = Dataset.from_dict(pref_data_train)
    pref_data = DatasetDict({'train': pref_data_train})

    # Save preference data
    pref_data.save_to_disk(args.scores_file.replace('scores', 'preference_data').replace('.txt', ''))


if __name__ == '__main__':
    main()
