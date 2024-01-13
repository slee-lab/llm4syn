#%%
import os
from os.path import join
import numpy as np
import torch
import json
import random
import math
from sklearn.model_selection import KFold  # Import KFold
from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
seedn=42
# random.seed(seedn)
from utils.data import *
from utils.model import *
device = 'cuda'
file_name = os.path.basename(__file__)
print("File Name:", file_name)

#%%
# login
from huggingface_hub import login
from env_config import *
login(hf_api_key_w, add_to_git_credential=True)
# wandb
import wandb
os.environ["WANDB_PROJECT"] = "syn_method" # name your W&B project 
# os.environ["WANDB_LOG_MODEL"] = "checkpoint" # log all model checkpoints
run_name="cavab"

# hparamm for training
overwrite_output_dir=True
nepochs = 200    # total eppochs for training 
num_folds = 5
ep_per_fold = nepochs//num_folds
lr=2e-5
wdecay=0.01
batch_size = 4
per_device_train_batch_size = batch_size  # default: 8
per_device_eval_batch_size = batch_size  # default: 8

print(f'total epochs: {nepochs}, kfolds: {num_folds}, epochs per fold: {ep_per_fold}')
print('learning rate: ', lr)
print('weight decay: ', wdecay)
print('per_device_train_batch_size: ', per_device_train_batch_size)
print('per_device_eval_batch_size: ', per_device_eval_batch_size)

#%%
# load data
random.seed(seedn)
sample_ratio = 1
data_path = '/home/rokabe/data2/cava/data/solid-state_dataset_2019-06-27_upd.json'  # path to the inorganic crystal synthesis data (json)
# data_path = '/home/rokabe/data2/cava/data/solutionsynthesis_dataset_202185.json'    # path to the solution based synthesis data (json)
data = json.load(open(data_path, 'r'))
num_sample = int(len(data)*sample_ratio)
separator=' || '
cut = ';'
rand_indices = random.sample(range(len(data)), num_sample)
data1 = [data[i] for i in rand_indices]
dataset = Dataset_Ceq2Ope(data1, index=None, te_ratio=0.1, separator=separator, cut=cut).dataset 
hf_model = "distilbert-base-uncased"
model_name = hf_usn + '/ope_bert_v1.4'# '/syn_distilgpt2_v2'
tk_model = hf_model #"Dagobert42/gpt2-finetuned-material-synthesis"#'m3rg-iitd/matscibert'##hf_model # set tokenizer model loaded from HF (usually same as hf_model)
load_pretrained=False   # If True, load the model from 'model_name'. Else, load the pre-trained model from hf_model. 
pad_tokenizer=False
save_indices = True
rm_ckpts = True

print('num data: ', num_sample)
print('hf_model: ', hf_model)
print('tk_model: ', tk_model)
print('model_name:', model_name)
#%%
# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(tk_model) 
if pad_tokenizer:
    tokenizer.pad_token = tokenizer.eos_token   #! false for bert model
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # checkk if we need this line. 
# Data collator
# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# test tokenizer
idx0 = 2
label = dataset['train'][idx0]['label']#[0]
print('label: ', label)
encoded_input0 = tokenizer(label)   # encoding  (label)
print('encoded (label): ', encoded_input0)
decoded_input0 = tokenizer.decode(encoded_input0["input_ids"])  # decoding  (label)
print('decoded (label): ', decoded_input0)
text = dataset['train'][idx0]['text']
print('text: ', text)
encoded_input1 = tokenizer(text)   ## encoding (text)
print('encoded (text): ', encoded_input1)
decoded_input1 = tokenizer.decode(encoded_input1["input_ids"])  # decoding  (text)
print('decoded (text): ', decoded_input1)

# def tokenize_function(examples):
#     return tokenizer(examples["text"], padding=True, truncation=True, return_tensors="pt") 
# tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names,)

def tokenize_function(examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result


# Use batched=True to activate fast multithreading!
tokenized_datasets = dataset.map(
    tokenize_function, batched=True, remove_columns=["text", "label"]
)
tokenized_datasets

#%%
# load model 
if load_pretrained:
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
else:      
    model = AutoModelForMaskedLM.from_pretrained(hf_model).to(device)   #!
model0 = AutoModelForMaskedLM.from_pretrained(hf_model).to(device)

distilbert_num_parameters = model.num_parameters() / 1_000_000
print(f"'>>> DistilBERT number of parameters: {round(distilbert_num_parameters)}M'")
print(f"'>>> BERT number of parameters: 110M'")

#%%
# refeernced from masked language model
text0 = "5 Si + 22 Li == 1 Li22Si5 || {'Mixing': None, 'ShapingOperation': None, 'HeatingOperation': {'heating_temperature': [{'max_value': 450.0, 'min_value': None, 'values': [[MASK]], 'units': 'C'}], 'heating_time': [{'max_value': 16.0, 'min_value': None, 'values': [16.0], 'units': 'h'}]}, 'QuenchingOperation': None}"
inputs = tokenizer(text0, return_tensors="pt").to(device)
token_logits = model(**inputs).logits
# Find the location of [MASK] and extract its logits
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
mask_token_logits = token_logits[0, mask_token_index, :]
# Pick the [MASK] candidates with the highest logits
top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

for token in top_5_tokens:
    print(f"'>>> {text0.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")

# Slicing produces a list of lists for each feature
tokenized_samples = tokenized_datasets["train"][:3]

for idx, sample in enumerate(tokenized_samples["input_ids"]):
    print(f"'>>> Review {idx} length: {len(sample)}'")

concatenated_examples = {
    k: sum(tokenized_samples[k], []) for k in tokenized_samples.keys()
}
total_length = len(concatenated_examples["input_ids"])
print(f"'>>> Concatenated reviews length: {total_length}'")

chunk_size = 128
chunks = {
    k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
    for k, t in concatenated_examples.items()
}

for chunk in chunks["input_ids"]:
    print(f"'>>> Chunk length: {len(chunk)}'")

#%%
# group text
def group_texts(examples):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(group_texts, batched=True)
lm_datasets

#%%
# data collator for masked language model
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

samples = [lm_datasets["train"][i] for i in range(2)]
for sample in samples:
    _ = sample.pop("word_ids")
# print(samples[0].keys())
for chunk in data_collator(samples)["input_ids"]:
    print(f"\n'>>> {tokenizer.decode(chunk)}'")

#%%
import collections
import numpy as np

from transformers import default_data_collator

wwm_probability = 0.2


def whole_word_masking_data_collator(features):
    for feature in features:
        word_ids = feature.pop("word_ids")

        # Create a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Randomly mask words
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        # print(mask) #!
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id
        feature["labels"] = new_labels

    return default_data_collator(features)


samples = [lm_datasets["train"][i] for i in range(2)]
batch = whole_word_masking_data_collator(samples)

for chunk in batch["input_ids"]:
    print(f"\n'>>> {tokenizer.decode(chunk)}'")

#%% 
# Inference using model before trainign before training 
# idx = 82
# data_source = 'test'
# out_type='add'
# out_size = 50
# remove_header=True
# print(idx)
# print('<<our prediction (before training)>>')
# output=show_one_test(model, dataset, idx, tokenizer, set_length={'type': out_type, 'value': out_size}, 
#                      separator=separator, remove_header=remove_header, source=data_source, device=device)
# print('<<Without training>>')
# output0=show_one_test(model0, dataset, idx, tokenizer, set_length={'type': out_type, 'value': out_size}, 
#                       separator=separator, remove_header=remove_header, source=data_source, device=device)


#%%
# Set up K-fold cross valudation
kf = KFold(n_splits=num_folds, shuffle=True, random_state=seedn)
ep_lists = get_epoch_lists(nepochs, num_folds, ep_per_fold)

logging_steps = len(lm_datasets["train"]) // batch_size
# model_name = model_checkpoint.split("/")[-1]

#%%
# training
epoch_count = 0
perplexity_scores = []
for i, ep_list in enumerate(ep_lists):
    for fold, (train_index, val_index) in enumerate(kf.split(lm_datasets['train'])):
        if fold < len(ep_list):
            print(f"Round {i}, Fold {fold + 1}/{num_folds}")

            epoch = ep_list[fold]
            # Create train and validation datasets for this fold
            # train_dataset = tokenized_datasets["train"].select(train_index)
            # val_dataset = tokenized_datasets["train"].select(val_index)
            train_dataset = lm_datasets["train"].select(train_index)
            val_dataset = lm_datasets["train"].select(val_index)
            training_args = TrainingArguments(
                output_dir=join('models', model_name),
                overwrite_output_dir=overwrite_output_dir,
                num_train_epochs = epoch,
                evaluation_strategy="epoch",
                learning_rate=lr,
                weight_decay=wdecay,
                push_to_hub=True,
                report_to="wandb",
                # run_name=f"{run_name}_i{i}_f{fold}", # name of the W&B run (optional)
                run_name=run_name, # name of the W&B run (optional)
                logging_steps=logging_steps,  #1  # how often to log to W&B
                per_device_train_batch_size=per_device_train_batch_size,
                per_device_eval_batch_size=per_device_eval_batch_size,
                save_total_limit=1, #!
                # load_best_model_at_end=True
                
            )
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator,
            )
            trainer.train()

            # model.push_to_hub(model_name)           
            eval_results = trainer.evaluate()
            perplexity = math.exp(eval_results['eval_loss'])
            print(f"Perplexity (Round {i}, Fold {fold + 1}): {perplexity:.2f}")
            # Store the perplexity score for this fold
            perplexity_scores.append(perplexity)
            epoch_count += epoch
            print('completed epochs: ', epoch_count)
            if fold==0:
                tokenizer.push_to_hub(model_name)   # save tokenizer to HF
    model.push_to_hub(model_name)
    # if i==0:
    #     tokenizer.push_to_hub(model_name)   # save tokenizer to HF

# Calculate and print the average perplexity score across all folds
avg_perplexity = np.mean(perplexity_scores)
std_perplexity = np.std(perplexity_scores)
print(f"Average Perplexity Across {num_folds} Folds: {avg_perplexity:.2f} (Std: {std_perplexity:.2f})")

if rm_ckpts:
    rm_files = join('models', model_name, '*')
    os.system(f'rm -r {rm_files}')  # delete the checkpoints, which are taking so large space. 
if save_indices:
    with open(f'./models/{model_name}/idx_s{sample_ratio}_seed{seedn}.txt', 'w') as f: 
        for i in rand_indices: f.write(f"{i}\n")

#%%
# print('test after training')
# idx = 9
# data_source = 'test'
# print(idx)
# print('<<our prediction (before training)>>')
# output=show_one_test(model, dataset, idx, tokenizer, set_length={'type': out_type, 'value': out_size}, 
#                      separator=separator, remove_header=remove_header, source=data_source, device=device)
# print('<<Without training>>')
# output0=show_one_test(model0, dataset, idx, tokenizer, set_length={'type': out_type, 'value': out_size}, 
#                       separator=separator, remove_header=remove_header, source=data_source, device=device)

#%%
def insert_random_mask(batch):
    features = [dict(zip(batch, t)) for t in zip(*batch.values())]
    masked_inputs = data_collator(features)
    # Create a new "masked" column for each column in the dataset
    return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}



# %%
