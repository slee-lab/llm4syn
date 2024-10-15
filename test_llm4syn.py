# Imports
import os
import random
import torch
from os.path import join
from transformers import AutoModelForCausalLM, set_seed
from huggingface_hub import login 
from env_config import hf_api_key_w, data_path, hf_load_name, seedn
from utils.data_config import separator_dict, out_conf_dict, gpt_model_dict, arrow_l2r
from utils.data import load_and_sample_data
from utils.model_utils import setup_tokenizer 
from utils.evaluate import evaluate_models

# Configuration
random.seed(seedn)
set_seed(seedn)
file_name = os.path.basename(__file__)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
login(hf_api_key_w, add_to_git_credential=True) 

# Hyperparameters
task = 'rhsope2lhs'  # choose one from ['lhs2rhs', 'rhs2lhs, 'lhsope2rhs', 'rhsope2lhs', 'tgt2ceq', 'tgtope2ceq']
model_tag = 'dgpt2' # pre-trained model tag
ver_tag = 'v1.2.1' # version of the model (make sure it is the same as the model you want to evaluate)
file_tag = '-1.1'   # file tag for saving the results. Leave as "" if you don't want to save the results
arrow = arrow_l2r 
separator, cut = separator_dict[task], ';'
run_name = f'{task}_{model_tag}_{ver_tag}'
model_name = join(hf_load_name, run_name)  
save_header = f'{run_name}{file_tag}' 
tk_model = model_name  
pad_tokenizer = True

# Load and sample dataset
dataset = load_and_sample_data(data_path, task, separator, te_ratio=0.1, cut=cut, arrow=arrow, sample_ratio=1)

# Initialize tokenizer
tokenizer = setup_tokenizer(tk_model, pad_tokenizer)

# Load models
model1 = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model0 = AutoModelForCausalLM.from_pretrained(gpt_model_dict[model_tag]).to(device)

model_dict = {'1': model1, '0': model0}    # 1: fine-tuned, 0: without fine-tuning

# Evaluate models
data_source = 'test'
num_sample = len(dataset[data_source])
adjust_gt_len = 0
gen_conf = {'num_beams':2, 'do_sample':True, 'num_beam_groups':1} 

df = evaluate_models(
                    model_dict=model_dict, 
                    dataset=dataset, 
                    tokenizer=tokenizer, 
                    num_sample=num_sample, 
                    header=save_header, 
                    data_source=data_source, 
                    gen_conf=gen_conf, 
                    adjust_gt_len=adjust_gt_len, 
                    separator=separator, 
                    set_length=out_conf_dict[task], 
                    device=device)


