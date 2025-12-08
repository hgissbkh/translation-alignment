import pandas as pd


MODELS_DICT = {
    'Base': {
        'short_name': 'base', 
        'full_name': 'ALMA-13B-SFT-HW', 
    },
    'Reference': {
        'short_name': 'ref',
    },
    'GPT-4': {
        'short_name': 'gpt4', 
    },
    'Rejected': {
        'short_name': 'rejected', 
    },
    'Chosen': {
        'short_name': 'chosen', 
    },
    'SFT-Multi-xCOMET-QE': {
        'short_name': 'sft_xcomet_multi',
        'full_name': 'ALMA-13B-SFT-HW-SFT-Multi-xCOMET-QE', 
    },
    'SFT-Multi-No-Ref-xCOMET-QE': {
        'short_name': 'sft_xcomet_multi_no_ref',
        'full_name': 'ALMA-13B-SFT-HW-SFT-Multi-No-Ref-xCOMET-QE', 
    },
    'SFT-Multi-No-Base-xCOMET-QE': {
        'short_name': 'sft_xcomet_multi_no_base',
        'full_name': 'ALMA-13B-SFT-HW-SFT-Multi-No-Base-xCOMET-QE', 
    },
    'SFT-Multi-No-GPT-4-xCOMET-QE': {
        'short_name': 'sft_xcomet_multi_no_gpt4',
        'full_name': 'ALMA-13B-SFT-HW-SFT-Multi-No-GPT-4-xCOMET-QE', 
    },
    'CPO-Multi-xCOMET-QE': {
        'short_name': 'cpo_xcomet_multi',
        'full_name': 'ALMA-13B-SFT-HW-CPO-Multi-xCOMET-QE',
    },
    'CPO-Multi-No-Ref-xCOMET-QE': {
        'short_name': 'cpo_xcomet_multi_no_ref',
        'full_name': 'ALMA-13B-SFT-HW-CPO-Multi-No-Ref-xCOMET-QE', 
    },
    'CPO-Multi-No-Base-xCOMET-QE': {
        'short_name': 'cpo_xcomet_multi_no_base',
        'full_name': 'ALMA-13B-SFT-HW-CPO-Multi-No-Base-xCOMET-QE', 
    },
    'CPO-Multi-No-GPT-4-xCOMET-QE': {
        'short_name': 'cpo_xcomet_multi_no_gpt4',
        'full_name': 'ALMA-13B-SFT-HW-CPO-Multi-No-GPT-4-xCOMET-QE', 
    },
    'SFT-Multi-CometKiwi': {
        'short_name': 'sft_kiwi_multi',
        'full_name': 'ALMA-13B-SFT-HW-SFT-Multi-CometKiwi', 
    },
    'CPO-Multi-CometKiwi': {
        'short_name': 'cpo_kiwi_multi',
        'full_name': 'ALMA-13B-SFT-HW-CPO-Multi-CometKiwi',
    },
    'SFT-Multi-chrF': {
        'short_name': 'sft_chrf_multi',
        'full_name': 'ALMA-13B-SFT-HW-SFT-Multi-chrF',
    },
    'CPO-Multi-chrF': {
        'short_name': 'cpo_chrf_multi',
        'full_name': 'ALMA-13B-SFT-HW-CPO-Multi-chrF',
    },
    'SFT-Multi-No-Ref-chrF': {
        'short_name': 'sft_chrf_multi_no_ref',
        'full_name': 'ALMA-13B-SFT-HW-SFT-Multi-No-Ref-chrF',
    },
    'CPO-Multi-No-Ref-chrF': {
        'short_name': 'cpo_chrf_multi_no_ref',
        'full_name': 'ALMA-13B-SFT-HW-CPO-Multi-No-Ref-chrF',
    },
    'SFT-Multi-xCOMET-QE-Choose-Base': {
        'short_name': 'sft_xcomet_multi_cb',
        'full_name': 'ALMA-13B-SFT-HW-SFT-Multi-xCOMET-QE-Choose-Base',
    },
    'CPO-Multi-xCOMET-QE-Choose-Base': {
        'short_name': 'cpo_xcomet_multi_cb',
        'full_name': 'ALMA-13B-SFT-HW-CPO-Multi-xCOMET-QE-Choose-Base',
    },
    'SFT-Multi-xCOMET-QE-Choose-GPT-4': {
        'short_name': 'sft_xcomet_multi_cg',
        'full_name': 'ALMA-13B-SFT-HW-SFT-Multi-xCOMET-QE-Choose-GPT-4',
    },
    'CPO-Multi-xCOMET-QE-Choose-GPT-4': {
        'short_name': 'cpo_xcomet_multi_cg',
        'full_name': 'ALMA-13B-SFT-HW-CPO-Multi-xCOMET-QE-Choose-GPT-4',
    },
    'SFT-Multi-xCOMET-QE-Choose-Ref': {
        'short_name': 'sft_xcomet_multi_cr',
        'full_name': 'ALMA-13B-SFT-HW-SFT-Multi-xCOMET-QE-Choose-Ref',
    },
    'CPO-Multi-xCOMET-QE-Choose-Ref': {
        'short_name': 'cpo_xcomet_multi_cr',
        'full_name': 'ALMA-13B-SFT-HW-CPO-Multi-xCOMET-QE-Choose-Ref',
    },
    'SFT-Multi-CometKiwi': {
        'short_name': 'sft_kiwi_multi',
        'full_name': 'ALMA-13B-SFT-HW-SFT-Multi-CometKiwi', 
    },
    'SFT-Mono-xCOMET-QE': {
        'short_name': 'sft_xcomet_mono',
        'full_name': 'ALMA-13B-SFT-HW-SFT-Mono-xCOMET-QE',
    }, 
    'CPO-Mono-xCOMET-QE': {
        'short_name': 'cpo_xcomet_mono',
        'full_name': 'ALMA-13B-SFT-HW-CPO-Mono-xCOMET-QE',
    }, 
    'CPO-Mono-xCOMET-QE-Choose-Low-Reject-Low': {
        'short_name': 'cpo_xcomet_mono_cl_rl',
        'full_name': 'ALMA-13B-SFT-HW-CPO-Mono-xCOMET-QE-Choose-Low-Reject-Low',
    },
    'CPO-Mono-xCOMET-QE-Choose-Low-Reject-Mid': {
        'short_name': 'cpo_xcomet_mono_cl_rm',
        'full_name': 'ALMA-13B-SFT-HW-CPO-Mono-xCOMET-QE-Choose-Low-Reject-Mid',
    },
    'CPO-Mono-xCOMET-QE-Choose-Low-Reject-High': {
        'short_name': 'cpo_xcomet_mono_cl_rh',
        'full_name': 'ALMA-13B-SFT-HW-CPO-Mono-xCOMET-QE-Choose-Low-Reject-High',
    },
    'CPO-Mono-xCOMET-QE-Choose-Mid-Reject-Low': {
        'short_name': 'cpo_xcomet_mono_cm_rl',
        'full_name': 'ALMA-13B-SFT-HW-CPO-Mono-xCOMET-QE-Choose-Mid-Reject-Low',
    },
    'CPO-Mono-xCOMET-QE-Choose-Mid-Reject-Mid': {
        'short_name': 'cpo_xcomet_mono_cm_rm',
        'full_name': 'ALMA-13B-SFT-HW-CPO-Mono-xCOMET-QE-Choose-Mid-Reject-Mid',
    },
    'CPO-Mono-xCOMET-QE-Choose-Mid-Reject-High': {
        'short_name': 'cpo_xcomet_mono_cm_rh',
        'full_name': 'ALMA-13B-SFT-HW-CPO-Mono-xCOMET-QE-Choose-Mid-Reject-High',
    },    
    'CPO-Mono-xCOMET-QE-Choose-High-Reject-Low': {
        'short_name': 'cpo_xcomet_mono_ch_rl',
        'full_name': 'ALMA-13B-SFT-HW-CPO-Mono-xCOMET-QE-Choose-High-Reject-Low',
    },
    'CPO-Mono-xCOMET-QE-Choose-High-Reject-Mid': {
        'short_name': 'cpo_xcomet_mono_ch_rm',
        'full_name': 'ALMA-13B-SFT-HW-CPO-Mono-xCOMET-QE-Choose-High-Reject-Mid',
    },
    'CPO-Mono-xCOMET-QE-Choose-High-Reject-High': {
        'short_name': 'cpo_xcomet_mono_ch_rh',
        'full_name': 'ALMA-13B-SFT-HW-CPO-Mono-xCOMET-QE-Choose-High-Reject-High',
    },
    'CPO-Mono-xCOMET-QE-Optimized': {
        'short_name': 'cpo_xcomet_mono_ch_rm',
        'full_name': 'ALMA-13B-SFT-HW-CPO-Mono-xCOMET-QE-Choose-High-Reject-Mid',
    },
}


METRICS_DICT = {
    'xCOMET-QE': 'xcomet',
    'CometKiwi': 'kiwi',
    'chrF': 'chrf',
    'Metric-X': 'metricx',
    'BLEU': 'bleu',
}


def make_table_eval(eval_df, table_info):
    # Initialize df
    columns = []
    for lp in table_info["lang_pairs"]:
        for metric in table_info["neural_metrics"]:
            columns.append((lp, "Neural", metric))
        columns.append((lp, "", ""))
        for metric in table_info["lexical_metrics"]:
            columns.append((lp, "Lexical", metric))
        columns.append(("", "", ""))
    df = pd.DataFrame(index=["Base"] + table_info["systems"], columns=pd.MultiIndex.from_tuples(columns[:-1]), data="")

    # Collect data
    for sys in ["Base"] + table_info["systems"]:
        for lp, typ, metric in df.columns: 
            if lp != "" and typ != "" and metric != "":
                if lp not in ["xx-en", "en-xx", "Avg."]:
                    df.loc[sys, (lp, typ, metric)] = eval_df.loc[eval_df.lp == lp, f"{models_dict[sys]['short_name']}_{metrics_dict[metric]}"].mean() 
                elif lp == "xx-en":
                    df.loc[sys, (lp, typ, metric)] = eval_df.loc[eval_df.tgt_lang == "en", f"{models_dict[sys]['short_name']}_{metrics_dict[metric]}"].mean()
                elif lp == "en-xx":
                    df.loc[sys, (lp, typ, metric)] = eval_df.loc[eval_df.src_lang == "en", f"{models_dict[sys]['short_name']}_{metrics_dict[metric]}"].mean()
                elif lp == "Avg.":
                    df.loc[sys, (lp, typ, metric)] = eval_df[f"{models_dict[sys]['short_name']}_{metrics_dict[metric]}"].mean()

    return df


def make_table_pref_data(table_info):
    # Initialize df
    index = []
    for dataset in table_info["datasets"]:
        if dataset == "ALMA-Preference-Multi":
            index += [(dataset, "Base"), (dataset, "GPT-4"), (dataset, "Reference")]
        else:
            index += [(dataset, "Rejected"), (dataset, "Chosen")]
    columns = []
    for lp in table_info["lang_pairs"]:
        for metric in table_info["neural_metrics"]:
            columns.append((lp, "Neural", metric))
        columns.append((lp, "", ""))
        for metric in table_info["lexical_metrics"]:
            columns.append((lp, "Lexical", metric))
        columns.append(("", "", ""))
    df = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(index),
        columns=pd.MultiIndex.from_tuples(columns)[:-1],
        data=""
    )

    # Collect data
    for ds in tqdm(table_info["datasets"]):
        pref_df = load_dataset(f"hgissbkh/{ds}")["train"].to_pandas()
        for sys in df.loc[ds].index:
            for lp, typ, metric in df.columns:
                if lp != "" and metric != "":
                    if lp not in ["xx-en", "en-xx", "Avg."]:
                        df.loc[(ds, sys), (lp, typ, metric)] = pref_df.loc[pref_df.lp == lp, f"{models_dict[sys]['short_name']}_{metrics_dict[metric]}"].mean() 
                    elif lp == "xx-en":
                        df.loc[(ds, sys), (lp, typ, metric)] = pref_df.loc[pref_df.tgt_lang == "en", f"{models_dict[sys]['short_name']}_{metrics_dict[metric]}"].mean()
                    elif lp == "en-xx":
                        df.loc[(ds, sys), (lp, typ, metric)] = pref_df.loc[pref_df.src_lang == "en", f"{models_dict[sys]['short_name']}_{metrics_dict[metric]}"].mean()
                    elif lp == "Avg.":
                        df.loc[(ds, sys), (lp, typ, metric)] = pref_df[f"{models_dict[sys]['short_name']}_{metrics_dict[metric]}"].mean()

    return df