#%%
import os
import numpy as np
import torch
import json
import random
import math
from utils.data import *
device = 'cuda'
file_name = os.path.basename(__file__)
print("File Name:", file_name)

#%%
# login
# RO
from huggingface_hub import login
from env_config import *
login(hf_api_key_w, add_to_git_credential=True)

#%%
# wandb
import wandb
os.environ["WANDB_PROJECT"] = "chemeq" # name your W&B project 
# os.environ["WANDB_LOG_MODEL"] = "checkpoint" # log all model checkpoints

#%%
# load data
data_path = '/home/rokabe/data2/cava/data/solid-state_dataset_2019-06-27_upd.json'
data = json.load(open(data_path, 'r'))
num_sample = int(len(data)*1)
# dataset = process_dataset(random.sample(data, num_sample), index=None, te_ratio=0.1, connector='|| ')
dataset = equation_dataset(random.sample(data, num_sample), index=None, te_ratio=0.1)
hf_model = "distilgpt2"     #"distilgpt2" #'pranav-s/MaterialsBERT'   #'Dagobert42/gpt2-finetuned-material-synthesis'   #'m3rg-iitd/matscibert'   #'HongyangLi/Matbert-finetuned-squad'
# model_name = 'RyotaroOKabe/chemeq_distilgpt2'
model_name = 'RyotaroOKabe/chemeq_distilgpt2'
load_pretrained=False
#%%
# load tokenizer
from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")    #TODO: explore other tokenizers!
# tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_7b")
# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# tokenizer = AutoTokenizer.from_pretrained("distilgpt2") # language_modeling.ipynb
tokenizer = AutoTokenizer.from_pretrained(hf_model) # language_modeling.ipynb #!
tokenizer.pad_token = tokenizer.eos_token   #!
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})    #!
# test tokenizer
## encoding (label)
prompt = dataset['train'][1]['label']#[0]
print('prompt: ', prompt)
encoded_input0 = tokenizer(prompt)
print('encoded (label): ', encoded_input0)
# decoding  (label)
decoded_input0 = tokenizer.decode(encoded_input0["input_ids"])
print('decoded (label): ', decoded_input0)
## encoding (equation)
equation = dataset['train'][1]['text']
print('equation: ', equation)
encoded_input1 = tokenizer(equation)
print('encoded (equation): ', encoded_input1)
# decoding  (equation)
decoded_input1 = tokenizer.decode(encoded_input1["input_ids"])
print('decoded (equation): ', decoded_input1)

# def tokenize_function(examples):
#     return tokenizer(examples["text"], padding="max_length", truncation=True, return_tensors="pt")
def tokenize_function(examples):
    return tokenizer(examples["text"], padding=True, truncation=True, return_tensors="pt")


tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names,)
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42)#.select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42)#.select(range(200))

#%%
# load model 
# from transformers import AutoModelForCausalLM

# model = AutoModelForCausalLM.from_pretrained(
#     "openlm-research/open_llama_7b", device_map="auto", load_in_4bit=True
# )
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

# model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)
# model0 = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)
if load_pretrained:
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
else:      
    model = AutoModelForCausalLM.from_pretrained(hf_model).to(device)   #!
model0 = AutoModelForCausalLM.from_pretrained(hf_model).to(device)

#%% before training 
prompt = dataset['train'][1]['label']#[0]
model_inputs = tokenizer(prompt, padding=True, truncation=True, return_tensors="pt")
model_inputs = {key: tensor.to(device) for key, tensor in model_inputs.items()}
# model_inputs['input_ids'] = torch.tensor([model_inputs['input_ids']])
# model_inputs['attention_mask'] = torch.tensor([model_inputs['attention_mask']])
print(model_inputs)
generated_ids = model.generate(**model_inputs)  #before training 
tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0] # before training 

#%%
# Data collator
from transformers import DataCollatorForLanguageModeling

# tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


#%% 
# Inference
idx = 110
data_source = 'test'
print(idx)
prompt = dataset[data_source][idx]['label']#[0]
model_inputs = tokenizer(prompt, padding=True, truncation=True, return_tensors="pt")
model_inputs = {key: tensor.to(device) for key, tensor in model_inputs.items()}
# model_inputs['input_ids'] = torch.tensor([model_inputs['input_ids']])
# model_inputs['attention_mask'] = torch.tensor([model_inputs['attention_mask']])
print(model_inputs)
generated_ids0 = model0.generate(**model_inputs)
generated_ids = model.generate(**model_inputs)
print('input: ', prompt)
encoded_input0 = tokenizer(prompt)
print('encoded (label): ', encoded_input0)
# decoding  (label)
decoded_input0 = tokenizer.decode(encoded_input0["input_ids"])
print('decoded (label): ', decoded_input0)
## encoding (equation)
equation = dataset[data_source][idx]['text']
print('equation: ', equation)
encoded_input1 = tokenizer(equation)
print('encoded (equation): ', encoded_input1)
# decoding  (equation)
decoded_input1 = tokenizer.decode(encoded_input1["input_ids"])
print('decoded (equation): ', decoded_input1)
output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
output0 = tokenizer.batch_decode(generated_ids0, skip_special_tokens=True)[0]
print('output (trained model):', output)
print('output (without training):', output0)
print(len(output), len(output0), generated_ids.shape, generated_ids0.shape, len(encoded_input0["input_ids"]))


#%%
# repeat the process
num_train = 1
for i in range(num_train):
    model_name_i = model_name + '_%.3d'%i
    print(f'Start {i}-th training')
    training_args = TrainingArguments(
        output_dir=model_name_i,#"chem_eq",
        num_train_epochs = 20,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        push_to_hub=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
    )

    trainer.train()

    model.push_to_hub(model_name_i)
    eval_results = trainer.evaluate()
    print(f"[{i}] Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    data_source = 'test'
    print(idx)
    prompt = dataset[data_source][idx]['label']#[0]
    model_inputs = tokenizer(prompt, padding=True, truncation=True, return_tensors="pt")
    model_inputs = {key: tensor.to(device) for key, tensor in model_inputs.items()}
    generated_ids = model.generate(**model_inputs)
    encoded_input0 = tokenizer(prompt)
    decoded_input0 = tokenizer.decode(encoded_input0["input_ids"])
    equation = dataset[data_source][idx]['text']
    print(f'[{i}] equation: ', equation)
    encoded_input1 = tokenizer(equation)
    decoded_input1 = tokenizer.decode(encoded_input1["input_ids"])
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f'[{i}] output (trained model):', output)
    print(len(output), generated_ids.shape, len(encoded_input0["input_ids"]))


#%%

































#%%
