def get_prompt(
        src, src_lang, tgt_lang, 
        template_type='Tower'
):
    lang_dict = {
        'de': 'German',
        'en': 'English',
        'es': 'Spanish',
        'fr': 'French',
        'it': 'Italian',
        'pt': 'Portuguese',
        'cs': 'Czech',
        'is': 'Icelandic',
        'ru': 'Russian',
        'zh': 'Chinese',
    }

    if template_type == 'Tower':
        template = '<|im_start|>user\nTranslate the following {SRC_LANG} source text to {TGT_LANG}:\n' + \
            '{SRC_LANG}: {SRC_SENTENCE}\n{TGT_LANG}: <|im_end|>\n<|im_start|>assistant\n'
    elif template_type == 'ALMA':
        template = 'Translate this from {SRC_LANG} into {TGT_LANG}:\n' + \
            '{SRC_LANG}: {SRC_SENTENCE}\n{TGT_LANG}:'
        
    prompt = template.format(
        SRC_SENTENCE=src, 
        SRC_LANG=lang_dict[src_lang], 
        TGT_LANG=lang_dict[tgt_lang]
    )
    return prompt