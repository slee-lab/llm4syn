#%%
import os
from os.path import join
import numpy as np
import torch
import json
import random
import math
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
seedn=42
# random.seed(seedn)
from utils.data import *
from utils.metrics import *
device = 'cuda'
file_name = os.path.basename(__file__)
print("File Name:", file_name)

from huggingface_hub import login
from env_config import *
login(hf_api_key_w, add_to_git_credential=True)

# import wandb
# os.environ["WANDB_PROJECT"] = "syn_method" # name your W&B project 

#%%
random.seed(seedn)
sample_ratio = 1
data_path = '/home/rokabe/data2/cava/data/solid-state_dataset_2019-06-27_upd.json'  # path to the inorganic crystal synthesis data (json)
# data_path = '/home/rokabe/data2/cava/data/solutionsynthesis_dataset_202185.json'    # path to the solution based synthesis data (json)
data = json.load(open(data_path, 'r'))
num_sample = int(len(data)*sample_ratio)
separator=' || '
cut = None #';'
rand_indices = random.sample(range(len(data)), num_sample)
data1 = [data[i] for i in rand_indices]
dataset = Dataset_Rhs2Lhs(data1, index=None, te_ratio=0.1, separator=separator, cut=cut).dataset 
run_name ='ceq_rl_mgpt_v1.2'
# hf_model = "gpt2" #"EleutherAI/gpt-neo-1.3B"   #"EleutherAI/gpt-j-6B"  #"distilgpt2"     #"distilgpt2" #'pranav-s/MaterialsBERT'   #'Dagobert42/gpt2-finetuned-material-synthesis'   #'m3rg-iitd/matscibert'   #'HongyangLi/Matbert-finetuned-squad'
model_name = join(hf_usn, run_name)    # '/ope_mgpt_v1.1' #'/tgt_mgpt_v1.4'
tk_model = model_name # set tokenizer model loaded from HF (usually same as hf_model)
load_pretrained=False   # If True, load the model from 'model_name'. Else, load the pre-trained model from hf_model. 
pad_tokenizer=True
save_indices = True

#%%
# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(tk_model) 
if pad_tokenizer:
    tokenizer.pad_token = tokenizer.eos_token   #!
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # checkk if we need this line. 

def tokenize_function(examples):
    return tokenizer(examples["text"], padding=True, truncation=True, return_tensors="pt")  # padding="max_length"

# tokenized dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names,)
small_train_dataset = tokenized_datasets["train"].shuffle(seed=seedn)
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=seedn)

#%%
# load model
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

#%%
# Inference using trained model 
idx = 40
data_source = 'test'  
out_type='mul'  # {'type': 'add', 'value': 50}, {'type': 'mul', 'value': 1.2}
out_size = 2.3 # 120, 2.5
remove_header=False
post_cut = ';'
print(idx)
print('<<our prediction>>')
output=show_one_test(model, dataset, idx, tokenizer, set_length={'type': out_type, 'value': out_size}, 
                     separator=separator, remove_header=remove_header, cut=post_cut, source=data_source, device=device)
print('gtruth: ', output['text']) 
print('answer: ', output['answer'])
# %%
# model view
from transformers import utils as t_utils
from bertviz import model_view, head_view
t_utils.logging.set_verbosity_error()  # Suppress standard warnings

input_text = output['answer']
model1 = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True).to(device)
inputs = tokenizer.encode(input_text, return_tensors='pt').to(device)  # Tokenize input text
outputs = model1(inputs)  # Run model
attention = outputs[-1]  # Retrieve attention from model outputs
tokens = tokenizer.convert_ids_to_tokens(inputs[0])  # Convert input ids to token strings

model_view(attention, tokens)  # Display model view
#%%
# head view
head_view(attention, tokens)
#%%
