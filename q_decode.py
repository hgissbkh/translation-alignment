import argparse
import pickle
import numpy as np
from datasets import load_dataset
from comet import download_model, load_from_checkpoint
from collections import Counter
from itertools import permutations


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-dn', '--dataset-name', type=str, default='mt-align-study-w-idiom-1203')
    parser.add_argument('-dp', '--dataset-path', type=str, default='data/raw/{DATASET_NAME}')
    parser.add_argument('-s', '--split', type=str, default='train')
    parser.add_argument('-ss', '--sub-split', type=str, default='None')
    parser.add_argument('-c', '--checkpoint', type=str, default='unaligned')
    parser.add_argument('-hp', '--hyperparameters', type=str, default='e3_bs32_lr1e-05_wu0.0')
    parser.add_argument('-mn', '--model-name', type=str, default='TowerInstruct-7B-v0.2')
    parser.add_argument('-cp', '--candidates-path', type=str, default='data/evaluation/{DATASET_NAME}/{MODEL_NAME}/{HYPERPARAMETERS}/{CHECKPOINT}')
    parser.add_argument('-cf', '--candidates-file', type=str, default='hypotheses_N20_t0.9_p0.6.txt')
    parser.add_argument('-dt', '--decoding-type', type=str, default='NBest')
    parser.add_argument('-scp', '--scorer-path', type=str, default='Unbabel/wmt22-comet-da')
    args = parser.parse_args()

    # Load data
    print('\n==========> Loading dataset...')
    dataset_path = args.dataset_path.format(DATASET_NAME=args.dataset_name)
    dataset = load_dataset(dataset_path)
    dataset = dataset[args.split]
    sources = dataset['src']

    # Load candidates
    if (args.checkpoint == 'unaligned') or (args.hyperparameters == 'None'):
        candidates_path = args.candidates_path.format(
            DATASET_NAME=args.dataset_name, 
            MODEL_NAME=args.model_name,
            HYPERPARAMETERS='',
            CHECKPOINT=args.checkpoint
        ).replace('//', '/')
    else:
        candidates_path = args.candidates_path.format(
            DATASET_NAME=args.dataset_name, 
            MODEL_NAME=args.model_name,
            HYPERPARAMETERS=args.hyperparameters,
            CHECKPOINT=args.checkpoint
        )
    num_candidates = int(args.candidates_file.split('_')[1][1:])
    with open(f'{candidates_path}/{args.candidates_file}', 'r') as f:
        candidates = [cd.replace('\\n', '\n') for cd in f.read().split('\n')]

    # Select data batch if relevant
    if args.sub_split != 'None':
        split_id, num_splits = args.sub_split.split('/')
        split_id, num_splits = int(split_id), int(num_splits)
        split_size_src = int(np.ceil(len(sources) / num_splits))
        sources = sources[(split_id-1) * split_size_src : split_id * split_size_src]
        split_size_cdt = split_size_src * num_candidates
        candidates = candidates[(split_id-1) * split_size_cdt : split_id * split_size_cdt]
    print('==========> Done.\n')
    
    # Load scorer
    print('==========> Loading scorer...')
    model = load_from_checkpoint(download_model(args.scorer_path))
    model.eval()
    model.cuda()
    score_name = args.scorer_path.split('/')[1].lower().replace('-', '_')
    print('==========> Done.\n')

    # Decode
    print('==========> Decoding...')    
    # N-Best reranking
    if args.decoding_type == 'N-Best':
        sources_rep = [src for src in sources for _ in range(num_candidates)]
        scoring_data = [{'src': src, 'mt': mt} for src, mt in zip(sources_rep, candidates)]
        scores = model.predict(scoring_data, gpus=1).scores
        
        # Retrieve best candidates and store QE scores
        best_candidates = []
        scoring_data = []
        for i in range(len(sources)):
            scs = scores[i * num_candidates : (i+1) * num_candidates] 
            cds = candidates[i * num_candidates : (i+1) * num_candidates]
            best_candidate = cds[np.argmax(scs)]
            best_candidates.append(best_candidate)
            scoring_data.append({'src': sources[i], 'mt': best_candidate, 'score': max(scs)})

        # Save scores as .pkl
        scores_file = args.candidates_file.replace('hypotheses', f'{score_name}_rf_nbest').replace('.txt', '.pkl')
        with open(f'{candidates_path}/{scores_file}', 'wb') as f:
            pickle.dump(scoring_data, f)

    # MBR decoding
    elif args.decoding_type == 'MBR':        
        # Get unique candidates and their frequencies
        cdts_counter = [Counter(candidates[i * num_candidates : (i+1) * num_candidates]) for i in range(len(sources))]
        cdts_unique = [list(counter.keys()) for counter in cdts_counter]
        cdts_freqs = [np.array(list(counter.values())) for counter in cdts_counter]

        # Format scoring data
        scoring_data = []
        for src, cdts in zip(sources, cdts_unique):
            pairs = list(permutations(cdts, 2))
            for cdt, pseudo_ref in pairs:
                scoring_data.append({'src': src, 'mt': cdt, 'ref': pseudo_ref})
        
        # Compute scores
        scores = np.array(model.predict(scoring_data, gpus=1).scores)

        # Retrieve best candidates
        best_candidates = []
        counter = 0
        for cdts, freqs in zip(cdts_unique, cdts_freqs):
            num_cdts = len(cdts)
            num_pairs = num_cdts * (num_cdts - 1)
            scs = scores[counter : counter + num_pairs]
            # Compute MBR matrix
            mbr_matrix = np.zeros((num_cdts, num_cdts))
            for i in range(len(mbr_matrix)):
                mbr_matrix[i, list(range(i)) + list(range(i+1, num_cdts))] = scs[i * (num_cdts - 1) : (i+1) * (num_cdts - 1)]
                mbr_matrix[i, i] = 1
            best_candidate = cdts[np.argmax(np.dot(mbr_matrix, freqs))]
            best_candidates.append(best_candidate)
            counter += num_pairs
    print('==========> Done.\n')

    # Set file name for saving
    translations_file = args.candidates_file.replace(
        'hypotheses', f'hypotheses_{args.decoding_type.lower().replace("-", "")}' 
        + f'_{args.sub_split.replace("/", "_")}' * (args.sub_split != 'None')
    )

    # Save hypotheses as .txt
    print('==========> Saving hypotheses...')
    with open(f'{candidates_path}/{translations_file}', 'w') as f:
        f.write('\n'.join([cd.replace('\n', '\\n') for cd in best_candidates]))
    print('==========> Done.\n')


if __name__ == '__main__':
    main()
