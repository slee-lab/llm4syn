#%%
"""
https://www.notion.so/240129-dataset-8e2ae778ed4b4c06830b4d9a5386a86c
"""
import os, sys
sys.path.append('../../')
from os.path import join
import numpy as np
import torch
import json
import random
import math
from sklearn.model_selection import KFold  # Import KFold
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
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
run_name="cava"

# hparamm for training
overwrite_output_dir=True
nepochs = 200    # total eppochs for training 
num_folds = 5
ep_per_fold = nepochs//num_folds
lr=2e-5
wdecay=0.01
per_device_train_batch_size = 4  # default: 8
per_device_eval_batch_size = per_device_train_batch_size  # default: 8

print(f'total epochs: {nepochs}, kfolds: {num_folds}, epochs per fold: {ep_per_fold}')
print('learning rate: ', lr)
print('weight decay: ', wdecay)
print('per_device_train_batch_size: ', per_device_train_batch_size)
print('per_device_eval_batch_size: ', per_device_eval_batch_size)

#%%
random.seed(seedn)
sample_ratio = 1
data_path = '/home/rokabe/data2/cava/data/solid-state_dataset_2019-06-27_upd.json'  # path to the inorganic crystal synthesis data (json)
# data_path = '/home/rokabe/data2/cava/data/solutionsynthesis_dataset_202185.json'    # path to the solution based synthesis data (json)
data = json.load(open(data_path, 'r'))

#%%
lhschems = []
ndata = len(data)//1
for i in range(ndata):
    lhs = data[i]['reaction']['left_side']
    for lh in lhs:
        mat = lh['material']
        if mat not in lhschems:
            lhschems.append(mat)

lhschems = sorted(lhschems)
print(f'reactions: {ndata}, #precursors: {len(lhschems)}')



# %%
