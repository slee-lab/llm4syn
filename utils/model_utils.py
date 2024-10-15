from transformers import AutoTokenizer


allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789()[]<>+-*=/., "

def get_epoch_lists(total_epochs, num_folds=5, ep_per_fold=1):
    # Initialize lists to store performance metrics for each fold
    epset = total_epochs//(num_folds*ep_per_fold)
    epset_frac = total_epochs%(num_folds*ep_per_fold)
    ep_lists = []
    for i in range(epset):
        ep_list = []
        for j in range(num_folds):
            ep_list.append(ep_per_fold)
        ep_lists.append(ep_list)
    total_sum = sum([sum(sublist) for sublist in ep_lists])
    if total_sum < total_epochs:
        ep_list = []
        while sum(ep_list)<epset_frac:
            ep_list.append(ep_per_fold)
        ep_lists.append(ep_list)

    print(ep_lists)
    total_sum = sum([sum(sublist) for sublist in ep_lists])
    print("Total epochs:", total_sum)
    return ep_lists

def setup_tokenizer(tk_model, pad_tokenizer):
    tokenizer = AutoTokenizer.from_pretrained(tk_model)
    if pad_tokenizer:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def tokenize_dataset(dataset, tokenizer, seedn):
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding=True, truncation=True, return_tensors="pt")
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)
    return tokenized_datasets["train"].shuffle(seed=seedn), tokenized_datasets["test"].shuffle(seed=seedn)

