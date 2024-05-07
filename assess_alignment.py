import argparse
import pickle
from tqdm import tqdm
import itertools
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
from sacrebleu.tokenizers import tokenizer_13a, tokenizer_zh


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-dn', '--dataset-name', type=str, default='wmt22-test')
    parser.add_argument('-dp', '--dataset-path', type=str, default='data/raw/{DATASET_NAME}')
    parser.add_argument('-s', '--split', type=str, default='test')
    parser.add_argument('-c', '--checkpoint', type=str, default='aligned')
    parser.add_argument('-mn', '--model-name', type=str, default='ALMA-13B')
    parser.add_argument('-hp', '--hyperparameters', type=str, default='None')
    parser.add_argument('-tp', '--translations-path', type=str, default='data/evaluation/{DATASET_NAME}/{MODEL_NAME}/{HYPERPARAMETERS}/{CHECKPOINT}')
    parser.add_argument('-tf', '--translations-file', type=str, default='hypotheses_greedy.txt')
    parser.add_argument('-ap', '--aligner-path', type=str, default='aneuraz/awesome-align-with-co')
    args = parser.parse_args()

    # Load data
    print('\n==========> Loading data...')
    dataset_path = args.dataset_path.format(DATASET_NAME=args.dataset_name)
    dataset = load_dataset(dataset_path)
    dataset = dataset[args.split]
    sources = dataset['src']
    lps = dataset['lp']
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

    # Load aligner
    print('==========> Loading aligner...')
    tokenizer = AutoTokenizer.from_pretrained(args.aligner_path)
    model = AutoModel.from_pretrained(args.aligner_path)
    print('==========> Done.\n')

    # Check word alignment
    print('==========> Assessing word alignment...')
    align_list = []   
    for src, tgt, lp in tqdm(list(zip(sources, translations, lps))):
        # Pre-process
        src_lang, tgt_lang = lp.split('-')
        if src_lang == 'zh':
            src = tokenizer_zh.TokenizerZh()(src)
        else:
            src = tokenizer_13a.Tokenizer13a()(src)
        if tgt_lang == 'zh':
            tgt = tokenizer_zh.TokenizerZh()(tgt)
        else:
            tgt = tokenizer_13a.Tokenizer13a()(tgt)
        sent_src, sent_tgt = src.strip().split(), tgt.strip().split()
        token_src, token_tgt = [tokenizer.tokenize(word) for word in sent_src], [tokenizer.tokenize(word) for word in sent_tgt]
        wid_src, wid_tgt = [tokenizer.convert_tokens_to_ids(x) for x in token_src], [tokenizer.convert_tokens_to_ids(x) for x in token_tgt]
        ids_src, ids_tgt = tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt', model_max_length=tokenizer.model_max_length, truncation=True)['input_ids'], tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt', truncation=True, model_max_length=tokenizer.model_max_length)['input_ids']
        sub2word_map_src = []
        for i, word_list in enumerate(token_src):
            sub2word_map_src += [i for x in word_list]
        sub2word_map_tgt = []
        for i, word_list in enumerate(token_tgt):
            sub2word_map_tgt += [i for x in word_list]

        # Align
        model.eval()
        with torch.no_grad():
            out_src = model(ids_src.unsqueeze(0), output_hidden_states=True)[2][8][0, 1:-1]
            out_tgt = model(ids_tgt.unsqueeze(0), output_hidden_states=True)[2][8][0, 1:-1]
            dot_prod = torch.matmul(out_src, out_tgt.transpose(-1, -2))
            softmax_srctgt = torch.nn.Softmax(dim=-1)(dot_prod)
            softmax_tgtsrc = torch.nn.Softmax(dim=-2)(dot_prod)
            softmax_inter = (softmax_srctgt > 1e-3) * (softmax_tgtsrc > 1e-3)
        align_subwords = torch.nonzero(softmax_inter, as_tuple=False)
        align_words = set()
        for i, j in align_subwords:
            align_words.add((sub2word_map_src[i], sub2word_map_tgt[j]))

        # Update align_list
        align_list.append({'src': src, 'mt': tgt, 'aligned_words': align_words})
    print('==========> Done.\n')

    # Save align_list as pickle
    print('==========> Saving scores...')
    align_file_name = args.translations_file.replace('hypotheses', 'aligned_words').replace('.txt', '.pkl')
    with open(f'{translations_path}/{align_file_name}', 'wb') as f:
        pickle.dump(align_list, f)
    print('==========> Done.\n') 


if __name__ == '__main__':
    main()