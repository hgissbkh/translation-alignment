# TranslationPO

This repository contains the resources for the paper **[Is Preference Alignment Always the Best Option to Enhance LLM-Based Translation? An Empirical Analysis](https://aclanthology.org/2024.wmt-1.127/)**, published at the Ninth Conference on Machine Translation (WMT24).

## Models and Datasets

All models and evaluation datasets used in the study are available in the following Hugging Face collection:  
<https://huggingface.co/collections/artefactory/translation-alignment-analysis>.

Model training was performed using the **ALMA** framework:  
<https://github.com/fe1ixxu/ALMA>.

## Reproducing Plots and Tables

To reproduce all plots and tables from the paper, run the notebook:  
[`plots_and_tables.ipynb`](plots_and_tables.ipynb).

## Citation

```bibtex
@inproceedings{gisserot-boukhlef-etal-2024-preference,
    title = "Is Preference Alignment Always the Best Option to Enhance {LLM}-Based Translation? An Empirical Analysis",
    author = "Gisserot-Boukhlef, Hippolyte  and
      Rei, Ricardo  and
      Malherbe, Emmanuel  and
      Hudelot, C{\'e}line  and
      Colombo, Pierre  and
      Guerreiro, Nuno M.",
    editor = "Haddow, Barry  and
      Kocmi, Tom  and
      Koehn, Philipp  and
      Monz, Christof",
    booktitle = "Proceedings of the Ninth Conference on Machine Translation",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.wmt-1.127/",
    doi = "10.18653/v1/2024.wmt-1.127",
    pages = "1373--1392",
    abstract = "Neural metrics for machine translation (MT) evaluation have become increasingly prominent due to their superior correlation with human judgments compared to traditional lexical metrics. Researchers have therefore utilized neural metrics through quality-informed decoding strategies, achieving better results than likelihood-based methods. With the rise of Large Language Models (LLMs), preference-based alignment techniques have gained attention for their potential to enhance translation quality by optimizing model weights directly on preferences induced by quality estimators. This study focuses on Contrastive Preference Optimization (CPO) and conducts extensive experiments to evaluate the impact of preference-based alignment on translation quality. Our findings indicate that while CPO consistently outperforms Supervised Fine-Tuning (SFT) on high-quality data with regard to the alignment metric, it may lead to instability across downstream evaluation metrics, particularly between neural and lexical ones. Additionally, we demonstrate that relying solely on the base model for generating candidate translations achieves performance comparable to using multiple external systems, while ensuring better consistency across downstream metrics."
}
```
