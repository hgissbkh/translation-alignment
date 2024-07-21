import argparse
import pickle
import numpy as np


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--dataset-path', type=str, default='data/raw/wmt22-23-test')
    parser.add_argument('-ms', '--model-state', type=str, default='unaligned')
    parser.add_argument('-cp', '--candidates-path', type=str, default='data/evaluation/{STATE}')
    parser.add_argument('-cf', '--candidates-file', type=str, default='hypotheses_N5_t0.25_p0.6.txt')
    args = parser.parse_args()

    # Load data
    print('\n==========> Loading dataset...')
    candidates_path = args.candidates_path.format(STATE=args.model_state)
    num_candidates = int(args.candidates_file.split('_')[1][1:])
    with open(f'{candidates_path}/{args.candidates_file}', 'r') as f:
        candidates = [cd.replace('\\n', '\n') for cd in f.read().split('\n')]
        candidates = [candidates[i * num_candidates : (i+1) * num_candidates] for i in range(int(len(candidates) / num_candidates))] 
    print('==========> Done.\n')
    
    # Load scores and compute kiwi-xcomet average
    print('==========> Loading scores...')
    scores_file_xcomet = args.candidates_file.replace('hypotheses', 'xcomet').replace('.txt', '.pkl')
    with open(f'{candidates_path}/{scores_file_xcomet}', 'rb') as f:
        cdt_scores_xcomet = pickle.load(f)
        cdt_scores_xcomet = [
            cdt_scores_xcomet[i * num_candidates : (i+1) * num_candidates] 
            for i in range(int(len(cdt_scores_xcomet) / num_candidates))
        ]
    scores_file_kiwi = args.candidates_file.replace('hypotheses', 'kiwi').replace('.txt', '.pkl')
    with open(f'{candidates_path}/{scores_file_kiwi}', 'rb') as f:
        cdt_scores_kiwi = pickle.load(f)
        cdt_scores_kiwi = [
            cdt_scores_kiwi[i * num_candidates : (i+1) * num_candidates] 
            for i in range(int(len(cdt_scores_kiwi) / num_candidates))
        ]
    cdt_scores = [
        [(xcomet + kiwi) / 2 for xcomet, kiwi in zip(xcomets, kiwis)] 
        for xcomets, kiwis in zip(cdt_scores_xcomet, cdt_scores_kiwi)
    ]
    print('==========> Done.\n')

    # Decode
    print('==========> Decoding...')    
    best_candidates, best_scores_kiwi, best_scores_xcomet = [], [], []
    for cdts, scs, kiwis, xcomets in zip(candidates, cdt_scores, cdt_scores_kiwi, cdt_scores_xcomet):
        sc_argmax = np.argmax(scs)
        best_candidates.append(cdts[sc_argmax])
        best_scores_kiwi.append(kiwis[sc_argmax])
        best_scores_xcomet.append(xcomets[sc_argmax])
    print('==========> Done.\n')

    # Save hypotheses and scores
    print('==========> Saving hypotheses and scores...')
    translations_file = args.candidates_file.replace('hypotheses', 'hypotheses_nbest')
    with open(f'{candidates_path}/{translations_file}', 'w') as f:
        f.write('\n'.join([cdt.replace('\n', '\\n') for cdt in best_candidates]))
    xcomet_file = translations_file.replace('hypotheses', 'xcomet').replace('.txt', '.pkl')
    with open(f'{candidates_path}/{xcomet_file}', 'wb') as f:
        pickle.dump(best_scores_xcomet, f)
    kiwi_file = translations_file.replace('hypotheses', 'kiwi').replace('.txt', '.pkl')
    with open(f'{candidates_path}/{kiwi_file}', 'wb') as f:
        pickle.dump(best_scores_kiwi, f)
    print('==========> Done.\n')


if __name__ == '__main__':
    main()
