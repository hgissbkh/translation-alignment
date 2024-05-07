import argparse
import pickle
from datasets import load_dataset
from comet import download_model, load_from_checkpoint


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-dn', '--dataset-name', type=str, default='mt-align-study-w-idiom-1203')
    parser.add_argument('-dp', '--dataset-path', type=str, default='data/raw/{DATASET_NAME}')
    parser.add_argument('-s', '--split', type=str, default='train')
    parser.add_argument('-c', '--checkpoint', type=str, default='aligned')
    parser.add_argument('-mn', '--model-name', type=str, default='TowerInstruct-7B-v0.2')
    parser.add_argument('-hp', '--hyperparameters', type=str, default='e3_bs32_lr1e-06_wu0.0')
    parser.add_argument('-tp', '--translations-path', type=str, default='data/evaluation/{DATASET_NAME}/{MODEL_NAME}/{HYPERPARAMETERS}/{CHECKPOINT}')
    parser.add_argument('-tf', '--translations-file', type=str, default='hypotheses_greedy.txt')
    parser.add_argument('-m', '--metric', type=str, default='COMET')
    parser.add_argument('-scp', '--scorer-path', type=str, default='Unbabel/wmt22-comet-da')
    parser.add_argument('-sm', '--scoring-mode', type=str, default='reference-based')
    args = parser.parse_args()

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
    lang_pairs = dataset['lp']
    if (args.checkpoint == 'unaligned') or (args.hyperparameters == 'None'):
        translations_path = args.translations_path.format(
            DATASET_NAME=args.dataset_name, 
            MODEL_NAME=args.model_name,
            HYPERPARAMETERS='',
            CHECKPOINT=args.checkpoint
        ).replace('//', '/')
    else:
        translations_path = args.translations_path.format(
            DATASET_NAME=args.dataset_name, 
            MODEL_NAME=args.model_name,
            HYPERPARAMETERS=args.hyperparameters,
            CHECKPOINT=args.checkpoint
        )
    with open(f'{translations_path}/{args.translations_file}', 'r') as f:
        translations = [tsl.replace('\\n', '\n') for tsl in f.read().split('\n')]
    print('==========> Done.\n')

    # Score translations with respect to references
    if args.metric == 'COMET':
        # Format data for scoring
        if args.scoring_mode == 'reference-based':
            scoring_data = [{'src': src, 'mt': cdt, 'ref': ref} for src, cdt, ref in zip(sources, translations, references)]
        elif args.scoring_mode == 'reference-free':
            scoring_data = [{'src': src, 'mt': cdt} for src, cdt in zip(sources, translations)]

        # Load scorer
        print('==========> Loading scorer...')
        model = load_from_checkpoint(download_model(args.scorer_path))
        print('==========> Done.\n')
        
        # Score hypotheses
        print('==========> Scoring hypotheses...')
        model_output = model.predict(scoring_data, gpus=1)
        scores = model_output.scores
        print('==========> Done.\n')
    else:
        raise NotImplementedError

    # Save scores
    print('==========> Saving scores...')
    score_name = args.scorer_path.split('/')[1].replace('-', '_').lower() \
        + '_rb' * (args.scoring_mode == 'reference-based') \
        + '_rf' * (args.scoring_mode == 'reference-free')
    scores_file = args.translations_file.replace('hypotheses', score_name).replace('.txt', '.pkl')  
    for elt, score, lp in zip(scoring_data, scores, lang_pairs):
        elt['score'] = score
        elt['lp'] = lp
    with open(f'{translations_path}/{scores_file}', 'wb') as f:
        pickle.dump(scoring_data, f)
    print('==========> Done.\n')


if __name__ == '__main__':
    main()
