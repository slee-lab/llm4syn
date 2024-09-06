#%%
import os
from os.path import join
import numpy as np
import torch
import pandas as pd
import pickle as pkl
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

# launch HF ans WandB
login(hf_api_key_w, add_to_git_credential=True)

#%%
# hparamm for training    #TODO save the config for wandb??
task = 'tgtope2ceq' # choose one from ['lhs2rhs', 'rhs2lhs, 'lhsope2rhs', 'rhsope2lhs', 'tgt2ceq', 'tgtope2ceq']
model_tag = 'dgpt2'
ver_tag = 'v1.2.1'

#%%
# [1] Load dataset
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
tk_model = model_name # set tokenizer model loaded from HF (usually same as hf_model)
pad_tokenizer=True
print('run_name: ', run_name)
print('model_name: ', model_name)

#%%
# [2] load tokenizer
tokenizer = setup_tokenizer(tk_model, pad_tokenizer)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding=True, truncation=True, return_tensors="pt")  # padding="max_length"

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names,)
small_train_dataset = tokenized_datasets["train"].shuffle(seed=seedn)
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=seedn)

#%%
# [3] load model
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model0 = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2").to(device)

#%%
# [4] Inference using trained model 
idx = 3
data_source = 'test'  
gen_conf = {'num_beams':6, 'do_sample':True, 'num_beam_groups':1}   #TODO: compare with different options?? 
print('gen_conf: ', gen_conf)
print(f'[{idx}] <<our prediction (before training)>>')
out_dict = one_result(model0, tokenizer, dataset, idx, set_length=out_conf_dict[task], 
                  separator=separator, source=data_source ,device='cuda', **gen_conf)
gt_text = out_dict['gt_text']
pr_text = out_dict['out_text']
print('gt_text: ', gt_text) 
print('pr_text: ', pr_text)

print(f'[{idx}] <<our prediction (after training)>>')
out_dict = one_result(model, tokenizer, dataset, idx, set_length=out_conf_dict[task], 
                separator=separator, source=data_source ,device='cuda', **gen_conf)
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



#%%
# [5] Plot element-wise prediction accuracy.
tag = 'v1.2'
num_sample = len(dataset[data_source])
sim_reacs, sim_prods, sim_all = [], [], []
lens_ceq, lens_opes = [], []
chem_dict = {el:[] for el in chemical_symbols}
df = pd.DataFrame(columns=['idx', 'label', 'gt_text', 'pr_text', 'gt_eq', 'pr_eq', 'acc'])
for idx in tqdm(range(num_sample), desc="Processing"):
    try:
        print(f'[{idx}] out_conf_dict[task]: ', out_conf_dict[task])
        out_dict = one_result(model, tokenizer, dataset, idx, set_length=out_conf_dict[task], 
                        separator=separator, source=data_source ,device='cuda', **gen_conf)
        gt_text = out_dict['gt_text']
        pr_text = out_dict['out_text']
        print('gt_text: ', gt_text) 
        print('pr_text: ', pr_text)
        gt_eq = gt_text.split(separator)[-1][1:]
        pr_eq = pr_text.split(separator)[-1][1:]
        label = out_dict['label']
        len_label = len(label)
        # ceq_gt, opes_gt = gt_text.split(separator)
        # len_ceq, len_opes = len(ceq_gt), len(opes_gt.split(' '))
        # lens_ceq.append(len_ceq)
        # lens_opes.append(len_opes)
        similarity_reactants, similarity_products, overall_similarity = equation_similarity(gt_eq, pr_eq, whole_equation=full_equation_dict[task], split='->')  #TODO: compare the different error functions??
        sim_reacs.append(similarity_reactants)
        sim_prods.append(similarity_products)
        sim_all.append(overall_similarity)
        print('acc: ', overall_similarity)
        label_elements = find_atomic_species(label)
        df = df._append({'idx': idx, 'label': label, 'gt_text': gt_text, 'pr_text': pr_text, 'gt_eq': gt_eq, 'pr_eq': pr_eq, 'acc': overall_similarity}, ignore_index=True)
        for el in label_elements:
            chem_dict[el].append(overall_similarity)
    except Exception as e:
        print(f"Error at idx={idx}: {e}")

header = run_name + '_' + data_source #'r2l_mean'
filename = f'./save/{header}_{num_sample}_{tag}.csv'

print(model_name)
print(f"(average) Reactants Similarity: {np.mean(sim_reacs):.2f}, Products Similarity: {np.mean(sim_prods):.2f}, Overall Similarity: {np.mean(sim_all):.2f}")
chem_mean_dict = {key: float(np.mean(value)) for key, value in chem_dict.items() if value}
header = run_name + '_' + data_source #'r2l_mean'
filename = f'./save/{header}_{num_sample}_{tag}.csv'
save_dict_as_csv(chem_mean_dict, filename)
print(f"Dictionary saved as {filename}")
# save chem_dict as pkl
with open(f'./save/{header}_{num_sample}_{tag}.pkl', 'wb') as f:
    pkl.dump(chem_dict, f)

from utils.periodic_trends import plotter
p = plotter(filename, output_filename=f'./save/{header}_{num_sample}_{tag}.html', under_value=0, over_value=1)


# fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# ax = axs[0]
# # ax.scatter(lens_tgt, sim_reacs, s=5, color='blue', label='Reactants')
# # ax.scatter(lens_tgt, sim_prods, s=5, color='red', label='Products')
# ax.scatter(lens_tgt, sim_all, s=5, color=good_colors['green'], label='Overall')
# ax.set_xlabel('Wrt target lengths', fontsize=14)
# ax.set_ylabel('Accuracy', fontsize=14)
# # ax.legend()

# ax = axs[1]
# # ax.scatter(lens_ceq, sim_reacs, s=5, color='blue', label='Reactants')
# # ax.scatter(lens_ceq, sim_prods, s=5, color='red', label='Products')
# ax.scatter(lens_ceq, sim_all, s=5, color=good_colors['green'], label='Overall')
# ax.set_xlabel('Wrt full equation lengths', fontsize=14)
# ax.set_ylabel('Accuracy', fontsize=14)
# # ax.legend()

# fig.suptitle(f'{header}_{num_sample}_{tag}', fontsize=16)
# fig.savefig(f'./save/{header}_{num_sample}_{tag}_scatter.png')

# len_data = np.array([lens_tgt, lens_ceq, sim_all]).T
# np.save(f'./save/{header}_{num_sample}_{tag}_len_data.npy', len_data)

# save df as csv 
df.to_csv(f'./save/{header}_{num_sample}_{tag}_df.csv')

# %%
# [6] model view (optional)
# from transformers import utils as t_utils
# from bertviz import model_view, head_view
# t_utils.logging.set_verbosity_error()  # Suppress standard warnings

# input_text = output['answer']
# model1 = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True).to(device)
# inputs = tokenizer.encode(input_text, return_tensors='pt').to(device)  # Tokenize input text
# outputs = model1(inputs)  # Run model
# attention = outputs[-1]  # Retrieve attention from model outputs
# tokens = tokenizer.convert_ids_to_tokens(inputs[0])  # Convert input ids to token strings

# model_view(attention, tokens)  # Display model view
# #%%
# # head view
# head_view(attention, tokens)
#%%
