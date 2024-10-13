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
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, set_seed
from transformers import DataCollatorForLanguageModeling
from huggingface_hub import login 
import wandb
from env_config import * 
from utils.data import *
from utils.model_utils import *
from utils.metrics import *
from utils.catalog import *
from utils.plot_data import good_colors
file_name = os.path.basename(__file__)

# launch HF ans WandB
login(hf_api_key_w, add_to_git_credential=True)

#%%
# hparamm for training    
task = 'rhs2lhs' # choose one from ['lhs2rhs', 'rhs2lhs, 'lhsope2rhs', 'rhsope2lhs', 'tgt2ceq', 'tgtope2ceq']
model_tag = 'dgpt2'
ver_tag = 'v1.2.1'
arrow = '->'    # '->', '==', 'etc. Make sure it is same as when we train the model. 

#%%
# [1] Load dataset
random.seed(seedn)
set_seed(seedn) #!
sample_ratio = 1
separator, cut = separator_dict[task], ';'
data = json.load(open(data_path, 'r'))
num_sample = int(len(data)*sample_ratio)
rand_indices = random.sample(range(len(data)), num_sample)
data1 = [data[i] for i in rand_indices]
dataset = Dataset_LLM4SYN(data1, index=None, te_ratio=0.1, separator=separator, cut=cut, arrow=arrow, task=task).dataset 
run_name = f'{task}_{model_tag}_{ver_tag}'  
model_name = join(hf_usn, run_name)   
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
gen_conf = {'num_beams':2, 'do_sample':True, 'num_beam_groups':1}   #TODO: compare with different options?? 
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
similarity_reactants, similarity_products, overall_similarity = equation_similarity(gt_eq, pr_eq, whole_equation=full_equation_dict[task], split=arrow) 
j_similarity = jaccard_similarity(gt_eq, pr_eq)

print(f"(average) Reactants Similarity: {similarity_reactants:.2f}, Products Similarity: {similarity_products:.2f}, Overall Similarity: {overall_similarity:.2f}")
print(f"Jaccard Similarity: {j_similarity:.2f}")

#%%
# load qweight from csv 
qw_file = './data/LLMSYNTHESIS_test_quantumweights.csv'
df_qw = pd.read_csv(qw_file)
# replace the keys in the column 'mp_id' with 'mpid', and 'quantum_weight' with 'qw'
df_qw = df_qw.rename(columns={'mp_id': 'mpid', 'quantum_weight': 'qw'})
mpids_qw = df_qw['mpid'].values

#%%
# [5] Plot element-wise prediction accuracy.
tag = 'qw'
num_sample = len(dataset[data_source])
sim_dict = {'reacts':[], 'prods':[], 'all':[], 'jac':[], 'lens_tgt':[], 'lens_ceq':[], 'len_opes':[], 'elem_tan':{el:[] for el in chemical_symbols}, 'elem_jac':{el:[] for el in chemical_symbols}}
sim_dict0 = {'reacts':[], 'prods':[], 'all':[], 'jac':[], 'lens_tgt':[], 'lens_ceq':[], 'len_opes':[], 'elem_tan':{el:[] for el in chemical_symbols}, 'elem_jac':{el:[] for el in chemical_symbols}}
model_list = [model, model0]
sim_dict_list = [sim_dict, sim_dict0]
label_list = ['', '.0']
gt_cut_add = 0
print('gt_cut_add: ', gt_cut_add)

df = pd.DataFrame(columns=['idx', 'mpid', 'formula', 'label', 'gt_text', 'pr_text', 'gt_eq', 'pr_eq', 'acc', 'jac', 'pr_text.0', 'pr_eq.0', 'acc.0', 'jac.0'])
for idx in tqdm(range(num_sample), desc="Processing"):
    row_dict = {}
    append_ = 0
    for model_, label_, sim_dict_ in zip(model_list, label_list, sim_dict_list):
        try:
            print(f'[{idx}] out_conf_dict[task]: ', out_conf_dict[task])
            out_dict = one_result(model_, tokenizer, dataset, idx, set_length=out_conf_dict[task], 
                            separator=separator, source=data_source ,device='cuda', **gen_conf) 
            label, gt_text, pr_text  = out_dict['label'], out_dict['gt_text'], out_dict['out_text']
            
            mpid = dataset[data_source][idx]['mpid']
            qw = df_qw.loc[df_qw['mpid'] == mpid, 'qw'].values[0] if mpid in mpids_qw else None
            formula = df_qw.loc[df_qw['mpid'] == mpid, 'formula'].values[0] if mpid in mpids_qw else None
            print(f'[{idx}] (mpid, qw): ', (mpid, qw))
            
            pr_text = pr_text.replace('\n', '')
            len_label = len(label)
            ###
            if gt_cut_add is not None:
                
                pr_text = pr_text[:int(len(gt_text)+gt_cut_add)]
            
            ###
            gt_eq = gt_text[len_label:]
            pr_eq = pr_text[len_label:]
            if len(pr_eq) == 0:
                pr_eq = 'Error'
                pr_text = pr_text + ' Error'
            similarity_reactants, similarity_products, overall_similarity = equation_similarity(gt_eq, pr_eq, whole_equation=full_equation_dict[task], split=arrow)  #TODO: compare the different error functions??
            j_similarity = jaccard_similarity(gt_eq, pr_eq)
            sim_dict_['reacts'].append(similarity_reactants)
            sim_dict_['prods'].append(similarity_products)
            sim_dict_['all'].append(overall_similarity)
            sim_dict_['jac'].append(j_similarity)
            print(f'gt_text{label_}: ', gt_text) 
            print(f'pr_text{label_}: ', pr_text)
            print(f'acc{label_}: ', overall_similarity)
            print(f"Jaccard Similarity{label_}: {overall_similarity:.2f}")
            label_elements = find_atomic_species(label)
            # append row_dict with the values
            row_dict.update({'idx': idx, 'mpid': mpid, 'formula': formula, 'qw': qw, f'label': label, f'gt_text': gt_text, f'pr_text{label_}': pr_text, f'gt_eq': gt_eq, f'pr_eq{label_}': pr_eq, f'acc{label_}': overall_similarity, f'jac{label_}': j_similarity})
            append_ += 1
            for el in label_elements:
                sim_dict_['elem_tan'][el].append(overall_similarity)
                sim_dict_['elem_jac'][el].append(j_similarity)
        except Exception as e:
            print(f"Error at idx={idx}: {e}")
            row_dict_ = {f'pr_text{label_}': label_keep+'Error', f'pr_eq{label_}': 'Error', f'acc{label_}': 0.0, f'jac{label_}': 0.0}
            print('row_dict_: ', row_dict_)
            row_dict.update(row_dict_ )
        label_keep = label
    if append_>0:
        df = df._append(row_dict, ignore_index=True)
    print('====================')


header = run_name + '_' + data_source #'r2l_mean'
# filename = f'./save/{header}_{num_sample}_{tag}.csv'

print(model_name)
for model_, label_, sim_dict_ in zip(model_list, label_list, sim_dict_list):
    print(f"(average) Reactants Similarity: {np.mean(sim_dict_['reacts']):.2f}, Products Similarity: {np.mean(sim_dict_['prods']):.2f}, Overall Similarity: {np.mean(sim_dict_['all']):.2f}")
    chem_mean_dict = {key: float(np.mean(value)) for key, value in sim_dict_['elem_tan'].items() if value}
    chem_mean_dict_jac = {key: float(np.mean(value)) for key, value in sim_dict_['elem_jac'].items() if value}
    header = run_name + '_' + data_source #'r2l_mean'
    filename = f'./save/{header}_{num_sample}_{tag}{label_}_tan.csv'
    save_dict_as_csv(chem_mean_dict, filename)
    filename_jac = f'./save/{header}_{num_sample}_{tag}{label_}_jac.csv'
    save_dict_as_csv(chem_mean_dict_jac, filename_jac)
    print(f"Dictionary saved as {filename}")
    # save chem_dict as pkl
    # with open(f'./save/{header}_{num_sample}_{tag}{label_}.pkl', 'wb') as f:
    #     pkl.dump(sim_dict_['elem_tan'], f)

    from utils.periodic_trends import plotter
    p = plotter(filename, output_filename=f'./save/{header}_{num_sample}_{tag}{label_}.html', under_value=0, over_value=1)

df.to_csv(f'./save/{header}_{num_sample}_{tag}_df.csv')


#%%
from matplotlib import pyplot as plt
plt.rcParams['font.family'] = ['Arial', 'sans-serif']

acc, qw, mpids, formulas = df['acc'], df['qw'], df['mpid'], df['formula']
# how many qw are not NaN
# print('Number of NaN in qw: ', qw.isnull().sum())
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.scatter(qw, acc, color=good_colors['blue'], alpha=0.5)
# add text to each point
for i, formula in enumerate(formulas):
    ax.text(qw[i], acc[i], formula, fontsize=8)
ax.set_title(f'[{task.upper()}] acc vs qw ({len(df)-qw.isnull().sum()}/{len(df)})')
ax.set_xlabel('$K_{xx}$')
ax.set_ylabel('GTS accuracy')
fig.savefig(f'./save/{header}_{num_sample}_{tag}_qw_vs_acc.pdf')

#%%
