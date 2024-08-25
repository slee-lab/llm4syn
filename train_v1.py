#%%
import os
from os.path import join
import numpy as np
import torch
import json
import random
import math
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
seedn=42
# random.seed(seedn)
from utils.data import *
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
os.environ["WANDB_PROJECT"] = "syn_method" # name your W&B project  #TODO add this to config file
# os.environ["WANDB_LOG_MODEL"] = "checkpoint" # log all model checkpoints
run_name="cava"  #TODO add this to config file

# hparam for training
overwrite_output_dir=True
nepochs = 50
lr=2e-5
wdecay=0.01
per_device_train_batch_size = 4  # default: 8
per_device_eval_batch_size = 4  # default: 8

print('epochs: ', nepochs)
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
separator=' == '
cut = ';'
rand_indices = random.sample(range(len(data)), num_sample)
data1 = [data[i] for i in rand_indices]
dataset = Dataset_Lhs2Rhs(data1, index=None, te_ratio=0.1, separator=separator, cut=cut).dataset 
hf_model =  "Dagobert42/gpt2-finetuned-material-synthesis" #"meta-llama/Llama-2-70b-chat-hf" #"EleutherAI/gpt-neo-1.3B"   #"EleutherAI/gpt-j-6B"  #"distilgpt2"     #"distilgpt2" #'pranav-s/MaterialsBERT'   #'Dagobert42/gpt2-finetuned-material-synthesis'   #'m3rg-iitd/matscibert'   #'HongyangLi/Matbert-finetuned-squad'
model_name = hf_usn + '/ceq_lr_mgpt_v1.3'# '/syn_distilgpt2_v2'
tk_model = "Dagobert42/gpt2-finetuned-material-synthesis"#'m3rg-iitd/matscibert'##hf_model # set tokenizer model loaded from HF (usually same as hf_model)
load_pretrained=False   # If True, load the model from 'model_name'. Else, load the pre-trained model from hf_model. 
pad_tokenizer=True
save_indices = True
rm_ckpts = True

print('num data: ', num_sample)
print('hf_model: ', hf_model)
print('model_name:', model_name)
#%%
# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(tk_model) 
if pad_tokenizer:
    tokenizer.pad_token = tokenizer.eos_token   #!
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # checkk if we need this line. 
# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# test tokenizer
idx0 = 2
label = dataset['train'][idx0]['label']#[0]
print('label: ', label)
encoded_input0 = tokenizer(label)   # encoding  (label)
print('encoded (label): ', encoded_input0)
decoded_input0 = tokenizer.decode(encoded_input0["input_ids"])  # decoding  (label)
print('decoded (label): ', decoded_input0)
text = dataset['train'][idx0]['text']
print('text: ', label)
encoded_input1 = tokenizer(label)   ## encoding (text)
print('encoded (text): ', encoded_input1)
decoded_input1 = tokenizer.decode(encoded_input1["input_ids"])  # decoding  (text)
print('decoded (text): ', decoded_input1)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding=True, truncation=True, return_tensors="pt")  # padding="max_length"

# tokenized dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names,)

#%%
# load model 
if load_pretrained:
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
else:      
    model = AutoModelForCausalLM.from_pretrained(hf_model).to(device)   #!
model0 = AutoModelForCausalLM.from_pretrained(hf_model).to(device)

#%% 
# Inference using model before trainign before training 
idx = 81
data_source = 'test'
out_type='add'
out_size = 50
remove_header=True
print(idx)
print('<<our prediction (before training)>>')
output=show_one_test(model, dataset, idx, tokenizer, set_length={'type': out_type, 'value': out_size}, 
                     separator=separator, remove_header=remove_header, source=data_source, device=device)
print('<<Without training>>')
output0=show_one_test(model0, dataset, idx, tokenizer, set_length={'type': out_type, 'value': out_size}, 
                      separator=separator, remove_header=remove_header, source=data_source, device=device)


#%%
# training
training_args = TrainingArguments(
    output_dir=join('models', model_name),
    overwrite_output_dir=overwrite_output_dir,
    num_train_epochs = nepochs,
    evaluation_strategy="epoch",
    learning_rate=lr,
    weight_decay=wdecay,
    push_to_hub=True,
    report_to="wandb",
    run_name=run_name,  # name of the W&B run (optional)
    logging_steps=1,   # how often to log to W&B
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
)

trainer.train()

model.push_to_hub(model_name)
tokenizer.push_to_hub(model_name)   # save tokenizer to HF
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

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
print(idx)
print('<<our prediction (before training)>>')
output=show_one_test(model, dataset, idx, tokenizer, set_length={'type': out_type, 'value': out_size}, 
                     separator=separator, remove_header=remove_header, source=data_source, device=device)
print('<<Without training>>')
output0=show_one_test(model0, dataset, idx, tokenizer, set_length={'type': out_type, 'value': out_size}, 
                      separator=separator, remove_header=remove_header, source=data_source, device=device)

# %%
