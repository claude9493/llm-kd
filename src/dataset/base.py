CUTOFF_LEN = 512

columns = ['input_ids', 'attention_mask', 'labels']

def generate_and_tokenize_prompt(instance, tokenizer, prompt_template, LABEL_SPLIT, is_test=False):
    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=CUTOFF_LEN,
            padding=True,
            return_tensors=None
        )
        if(
            result['input_ids'][-1] != tokenizer.eos_token_id
            and len(result['input_ids']) < CUTOFF_LEN
            and add_eos_token
        ):
            result['input_ids'].append(tokenizer.eos_token_id)
            result['attention_mask'].append(1)
        result['labels'] = result['input_ids'].copy()
        return result
    tokenized_full_prompt = tokenize(prompt_template.format(**instance))
    tokenized_user_prompt = tokenize(prompt_template.format(**instance).split(LABEL_SPLIT)[0] + LABEL_SPLIT, add_eos_token=False)
    user_prompt_len = len(tokenized_user_prompt['input_ids'])
    tokenized_full_prompt['labels'] = [-100]*user_prompt_len + tokenized_full_prompt['labels'][user_prompt_len:]
    if is_test:
        tokenized_user_prompt['_id'] = instance['id']
        return tokenized_user_prompt
    
    len_labels = len(tokenizer(instance['summary'])['input_ids'])
    tokenized_full_prompt['is_label_complete'] = len(tokenized_full_prompt['labels'][user_prompt_len:]) >= len_labels
    return tokenized_full_prompt

