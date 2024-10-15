#%%
import os
from os.path import join
import numpy as np
import torch
import random
import math
from sklearn.model_selection import KFold
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, set_seed
from huggingface_hub import login
import wandb
from env_config import hf_api_key_w, data_path, hf_usn, wandb_project, seedn
from utils.data_config import separator_dict, gpt_model_dict, arrow_l2r
from utils.data import load_and_sample_data
from utils.model_utils import setup_tokenizer, tokenize_dataset, get_epoch_lists
from utils.utilities import make_dict

# Configuration
random.seed(seedn)
set_seed(seedn)
file_name = os.path.basename(__file__)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'{file_name=}')
login(hf_api_key_w, add_to_git_credential=True)
os.environ["WANDB_PROJECT"] = wandb_project 

#%%
# Hyperparameters
# general
task = 'tgt2ceq' # choose one from ['lhs2rhs', 'rhs2lhs, 'lhsope2rhs', 'rhsope2lhs', 'tgt2ceq', 'tgtope2ceq']
model_tag = 'dgpt2' # pre-trained model tag
ver_tag = 'v1.X'  # version of the model
arrow = arrow_l2r 

# training
overwrite_output_dir=True
nepochs = 100    # total epochs for training 
num_folds = 10
ep_per_fold = nepochs//num_folds
lr=2e-5
wdecay=0.01
per_device_train_batch_size = 4  # default: 8
per_device_eval_batch_size = per_device_train_batch_size 
load_pretrained=False
pad_tokenizer=True
save_indices = True
rm_ckpts = True
separator, cut = separator_dict[task], ';'
run_name = f'{task}_{model_tag}_{ver_tag}' 
model_name = join(hf_usn, run_name) 
hf_model = gpt_model_dict[model_tag] 
tk_model = hf_model


#%%
# load data
sample_ratio = 1
dataset = load_and_sample_data(data_path, task, separator, te_ratio=0.1, cut=cut, arrow=arrow, sample_ratio=sample_ratio, save_idx_name=model_name)

conf_dict = make_dict([
    file_name, arrow, separator, cut, nepochs, num_folds, ep_per_fold, lr, wdecay, 
    per_device_train_batch_size, per_device_eval_batch_size,
    run_name, hf_model, model_name, tk_model, load_pretrained, 
    pad_tokenizer, save_indices, rm_ckpts
])
for key, val in conf_dict.items():
    print(f'{key}: {val}')
#%%
# Tokenizer and data collation
tokenizer = setup_tokenizer(tk_model, pad_tokenizer)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
train_dataset, test_dataset = tokenize_dataset(dataset, tokenizer, seedn)

#%%
# Model loading
if load_pretrained:
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
else:      
    model = AutoModelForCausalLM.from_pretrained(hf_model).to(device)  
model0 = AutoModelForCausalLM.from_pretrained(hf_model).to(device)
print(model.config) 
    
#%%
# K-Fold cross-validation
kf = KFold(n_splits=num_folds, shuffle=True, random_state=seedn)
ep_lists = get_epoch_lists(nepochs, num_folds, ep_per_fold)
print(f'{ep_lists=}')

#%%
# Ttraining  
epoch_count = 0
perplexity_scores = []
for i, ep_list in enumerate(ep_lists):
    for fold, (train_index, val_index) in enumerate(kf.split(dataset['train'])):
        if fold >= len(ep_list):  
            continue
        print(f"Round {i}, Fold {fold + 1}/{num_folds}")

        epoch = ep_list[fold]
        # Create train and validation datasets for this fold
        train_dataset, val_dataset = train_dataset.select(train_index), train_dataset.select(val_index)
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
            save_total_limit=1, 
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
        print('eval_results: ', eval_results)
        perplexity = math.exp(eval_results['eval_loss'])
        print(f"Perplexity (Round {i}, Fold {fold + 1}): {perplexity:.2f}") 
        # Store the perplexity score for this fold
        perplexity_scores.append(perplexity)
        epoch_count += epoch
        print('completed epochs: ', epoch_count)
        if i==0 and fold==0:
            tokenizer.push_to_hub(model_name)   # save tokenizer to HF
            wandb.config.update(conf_dict) 
        wandb.log({'perplexity': perplexity, 'epoch_count': epoch_count})  
    model.push_to_hub(model_name)

# Calculate and print the average perplexity score across all folds
avg_perplexity = np.mean(perplexity_scores)
std_perplexity = np.std(perplexity_scores)
print(f"Average Perplexity Across {num_folds} Folds: {avg_perplexity:.2f} (Std: {std_perplexity:.2f})")

if rm_ckpts:
    rm_files = join('models', model_name, '*')
    os.system(f'rm -r {rm_files}')  # delete the checkpoints, which are taking so large space. 


# %%
