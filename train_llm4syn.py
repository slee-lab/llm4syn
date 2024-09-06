#%%
import os
from os.path import join
import numpy as np
import torch
import json
import random
import math
from sklearn.model_selection import KFold 
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from huggingface_hub import login 
import wandb
from env_config import * 
from utils.data import *
from utils.model_utils import *
from utils.metrics import *
from utils.catalog import *
file_name = os.path.basename(__file__)
# use f-string for all print statements
print(f'{file_name=}')

#%%
# launch HF ans WandB
login(hf_api_key_w, add_to_git_credential=True)
os.environ["WANDB_PROJECT"] = wandb_project 
# os.environ["WANDB_LOG_MODEL"] = "checkpoint" # log all model checkpoints

# hparamm for training    #TODO save the config for wandb??
task = 'tgt2ceq' # choose one from ['lhs2rhs', 'rhs2lhs, 'lhsope2rhs', 'rhsope2lhs', 'tgt2ceq', 'tgtope2ceq']
model_tag = 'dgpt2'
ver_tag = 'v1.2.1'

overwrite_output_dir=True
nepochs = 100    # total eppochs for training 
num_folds = 10
ep_per_fold = nepochs//num_folds
lr=2e-5
wdecay=0.01
per_device_train_batch_size = 4  # default: 8
per_device_eval_batch_size = per_device_train_batch_size  # default: 8
load_pretrained=True
pad_tokenizer=True
save_indices = True
rm_ckpts = True


#%%
# load data
random.seed(seedn)
sample_ratio = 1
separator, cut = separator_dict[task], ';'
data = json.load(open(data_path, 'r'))
num_sample = int(len(data)*sample_ratio)
rand_indices = random.sample(range(len(data)), num_sample)
data1 = [data[i] for i in rand_indices]
dataset = Dataset_LLM4SYN(data1, index=None, te_ratio=0.1, separator=separator, cut=cut, task=task).dataset 
run_name = f'{task}_{model_tag}_{ver_tag}'  #TODO put all config part into one place
model_name = join(hf_usn, run_name)   #TODO any newer model? 
hf_model = gpt_model_dict[model_tag] 
tk_model = hf_model


conf_dict = make_dict([
    file_name, separator, cut, nepochs, num_folds, ep_per_fold, lr, wdecay, 
    per_device_train_batch_size, per_device_eval_batch_size,
    run_name, hf_model, model_name, tk_model, load_pretrained, 
    pad_tokenizer, save_indices, rm_ckpts
])
print(conf_dict)
#%%
# load tokenizer
tokenizer = setup_tokenizer(tk_model, pad_tokenizer)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# test tokenizer    #TODO skip this section in the end??
idx0 = 2

try:
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
except Exception as e:
    print('error: ', e)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding=True, truncation=True, return_tensors="pt")  # padding="max_length"

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names,)

#%%
# load model 
# load model 
if load_pretrained:
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
else:      
    model = AutoModelForCausalLM.from_pretrained(hf_model).to(device)   #!
model0 = AutoModelForCausalLM.from_pretrained(hf_model).to(device)
print(model.config) 

#%% 
# Inference using model before trainign before training #TODOdelete this section in the end?? 
idx = 82
try: 
    data_source = 'test'
    print(f'[{idx}] <<our prediction (before training)>>')
    out_dict = one_result(model0, tokenizer, dataset, idx, set_length=out_conf_dict[task], 
                    separator=separator, source=data_source ,device='cuda')
    print('gt_text: ', out_dict['gt_text']) 
    print('out_text: ', out_dict['out_text'])

    print(f'[{idx}] <<our prediction (after training)>>')
    out_dict = one_result(model, tokenizer, dataset, idx, set_length=out_conf_dict[task], 
                    separator=separator, source=data_source ,device='cuda')
    gt_text = out_dict['gt_text']
    pr_text = out_dict['out_text']
    print('gt_text: ', gt_text) 
    print('pr_text: ', pr_text)
    gt_eq = gt_text.split(separator)[-1][1:]
    pr_eq = pr_text.split(separator)[-1][1:]

    print('eq_gt: ', gt_eq)
    print('eq_pred: ', pr_eq)
    similarity_reactants, similarity_products, overall_similarity = equation_similarity(gt_eq, pr_eq, whole_equation=full_equation_dict[task], split='->') 
    print(f"(average) Reactants Similarity: {similarity_reactants:.2f}, Products Similarity: {similarity_products:.2f}, Overall Similarity: {overall_similarity:.2f}")
except Exception as e:
    print('error: ', e)
    
#%%
# Set up K-fold cross valudation
kf = KFold(n_splits=num_folds, shuffle=True, random_state=seedn)
ep_lists = get_epoch_lists(nepochs, num_folds, ep_per_fold)
print(f'{ep_lists=}')

#%%
# training  #TODO: can we make this part more concise??
epoch_count = 0
perplexity_scores = []
for i, ep_list in enumerate(ep_lists):
    for fold, (train_index, val_index) in enumerate(kf.split(dataset['train'])):
        if fold >= len(ep_list):    #! if the number of epochs is not enough for the number of folds,
            continue
        print(f"Round {i}, Fold {fold + 1}/{num_folds}")

        epoch = ep_list[fold]
        # Create train and validation datasets for this fold
        train_dataset = tokenized_datasets["train"].select(train_index)
        val_dataset = tokenized_datasets["train"].select(val_index)
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
            logging_steps=1,   # how often to log to W&B
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            save_total_limit=1, #!
            # load_best_model_at_end=True
            
        )
        trainer = Trainer(    #! CustomTrainer instead of Trainer
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        trainer.train()

        # model.push_to_hub(model_name)           
        eval_results = trainer.evaluate()
        print('eval_results: ', eval_results)
        perplexity = math.exp(eval_results['eval_loss'])
        print(f"Perplexity (Round {i}, Fold {fold + 1}): {perplexity:.2f}") #TODO could e present perplexity in the paper?? 
        # Store the perplexity score for this fold
        perplexity_scores.append(perplexity)
        epoch_count += epoch
        print('completed epochs: ', epoch_count)
        if fold==0:
            tokenizer.push_to_hub(model_name)   # save tokenizer to HF
            wandb.config.update(conf_dict)  #! update the config for wandb
        wandb.log({'perplexity': perplexity, 'epoch_count': epoch_count})   #!
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
print('test after training')
idx = 9
data_source = 'test'
print(f'[{idx}] <<our prediction (before training)>>')
out_dict = one_result(model0, tokenizer, dataset, idx, set_length=out_conf_dict[task], 
                  separator=separator, source=data_source ,device='cuda')
print('gt_text: ', out_dict['gt_text']) 
print('out_text: ', out_dict['out_text'])

print(f'[{idx}] <<our prediction (after training)>>')
out_dict = one_result(model, tokenizer, dataset, idx, set_length=out_conf_dict[task], 
                  separator=separator, source=data_source ,device='cuda')
print('gt_text: ', out_dict['gt_text']) 
print('out_text: ', out_dict['out_text'])

# %%
