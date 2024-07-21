import argparse
import os
import pickle
from tqdm import tqdm
import itertools
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
from sacrebleu.tokenizers import tokenizer_13a, tokenizer_zh


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf-dataset-path', type=str, default='hgissbkh/wmt22-23-test')
    parser.add_argument('--hf-model-path', type=str, default='hgissbkh/ALMA-13B-SFT-HW')
    parser.add_argument('--stage', type=str, default='evaluation')
    parser.add_argument('--load-save-path', type=str, default='data/{STAGE}/{DATASET_NAME}/{MODEL_NAME}')
    parser.add_argument('--translations-file', type=str, default='hypotheses_greedy.txt')
    parser.add_argument('--aligner-path', type=str, default='aneuraz/awesome-align-with-co')
    args = parser.parse_args()

    # Load data
    print('\n==========> Loading data...')
    if args.stage == "training":
        dataset = load_dataset(args.hf_dataset_path)["train"]
        sources = dataset["src"]
        lps = dataset["lp"]
        translations = [{"rejected": ex["rejected"], "chosen": ex["chosen"], "alma": ex["alma"]} for ex in dataset]
    elif args.stage == "evaluation":
        dataset = load_dataset(args.hf_dataset_path)["train"]
        sources = dataset["src"]
        lps = dataset["lp"]
        translations_path = args.load_save_path.format(
            STAGE=args.stage, 
            DATASET_NAME=args.hf_dataset_path.split("/")[-1],
            MODEL_NAME=args.hf_model_path.split("/")[-1]
        )
        with open(f"{translations_path}/{args.translations_file}", "r") as f:
            translations = [{"translation": tsl.replace("\\n", "\n")} for tsl in f.read().split("\n")]
    print('==========> Done.\n')

    # Load aligner
    print('==========> Loading aligner...')
    tokenizer = AutoTokenizer.from_pretrained(args.aligner_path)
    model = AutoModel.from_pretrained(args.aligner_path)
    print('==========> Done.\n')

    # Check word alignment
    print('==========> Assessing word alignment...')
    align_list = []   
    for src, tsl, lp in tqdm(list(zip(sources, translations, lps))):
        src_lang, tgt_lang = lp.split('-')    
        if src_lang == 'zh':
            src = tokenizer_zh.TokenizerZh()(src)
        else:
            src = tokenizer_13a.Tokenizer13a()(src)
        if tgt_lang == 'zh':
            tok_tgt = tokenizer_zh.TokenizerZh()
        else:
            tok_tgt = tokenizer_13a.Tokenizer13a()
        align_dict = tsl.copy()

        for k, tgt in tsl.items():
            tgt = tok_tgt(tgt)
            sent_src, sent_tgt = src.strip().split(), tgt.strip().split()
            token_src, token_tgt = [tokenizer.tokenize(word) for word in sent_src], [tokenizer.tokenize(word) for word in sent_tgt]
            wid_src, wid_tgt = [tokenizer.convert_tokens_to_ids(x) for x in token_src], [tokenizer.convert_tokens_to_ids(x) for x in token_tgt]
            ids_src, ids_tgt = tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt', model_max_length=tokenizer.model_max_length, truncation=True)['input_ids'], tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt', truncation=True, model_max_length=tokenizer.model_max_length)['input_ids']
            sub2word_map_src = []
            for i, word_list in enumerate(token_src):
                sub2word_map_src += [i for _ in word_list]
            sub2word_map_tgt = []
            for i, word_list in enumerate(token_tgt):
                sub2word_map_tgt += [i for _ in word_list]
            model.eval()
            with torch.no_grad():
                out_src = model(ids_src.unsqueeze(0), output_hidden_states=True)[2][8][0, 1:-1]
                out_tgt = model(ids_tgt.unsqueeze(0), output_hidden_states=True)[2][8][0, 1:-1]
                dot_prod = torch.matmul(out_src, out_tgt.transpose(-1, -2))
            align_df = pd.DataFrame(
                dot_prod.numpy(), 
                index=sub2word_map_src[:dot_prod.shape[0]], 
                columns=sub2word_map_tgt[:dot_prod.shape[1]]
            )
            align_dict[k] = align_df
        align_list.append(align_dict)
    print('==========> Done.\n')
    
    # Save align_list as pickle
    print('==========> Saving scores...')  
    if args.stage == 'training':
        save_path = args.load_save_path.format(
            STAGE=args.stage, 
            DATASET_NAME=args.hf_dataset_path.split("/")[-1],
            MODEL_NAME=""
        )[:-1]
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(f"{save_path}/word_alignment.pkl", "wb") as f:
            pickle.dump(align_list, f)
    elif args.stage == "evaluation":    
        align_file_name = args.translations_file.replace('hypotheses', 'word_alignment').replace('.txt', '.pkl')
        with open(f'{translations_path}/{align_file_name}', 'wb') as f:
            pickle.dump(align_list, f)
    print('==========> Done.\n') 


if __name__ == '__main__':
    main()