def get_prompt(
        src, src_lang, tgt_lang, 
        template='<|im_start|>user\nTranslate the following [SRC_LANG] source text to [TGT_LANG]:\n' + \
            '[SRC_LANG]: [SRC_SENTENCE]\n[TGT_LANG]: <|im_end|>\n<|im_start|>assistant\n'
):
    lang_dict = {
        'de': 'German',
        'en': 'English',
        'es': 'Spanish',
        'fr': 'French',
        'it': 'Italian',
        'pt': 'Portuguese'
    }
    prompt = template.replace(
        '[SRC_SENTENCE]', src
    ).replace(
        '[SRC_LANG]', lang_dict[src_lang]
    ).replace(
        '[TGT_LANG]', lang_dict[tgt_lang]
    )
    return prompt