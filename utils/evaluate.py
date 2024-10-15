from os.path import join
import re
import csv
import pandas as pd
import random
from tqdm import tqdm
from utils.utilities import chemical_symbols
from utils.data_config import separator_out
from env_config import save_dir

black_list = ['?', '!', ';', '{', '}', '\\', '@', '#', '$', '%', '^', '&', '_', '~', '`', 'δ', '�', 'ㅋ']


def find_atomic_species(formula):   #TODO: could we combine this with the element_vector function?. Also check if ''fformula' is correct. 
    # Regular expression pattern for element symbols: one uppercase letter followed by an optional lowercase letter
    pattern = r'[A-Z][a-z]?'
    
    # Find all occurrences of the pattern in the formula string
    elements = re.findall(pattern, formula)
    
    # Remove duplicates by converting the list to a set, then convert back to a list if needed
    unique_elements = list(set(elements))
    # print('unieuq_elements: ', unique_elements)
    chem_list = []
    for el in unique_elements:
        if el in chemical_symbols:
            chem_list.append(el)
        else: 
            if el[0] in chemical_symbols:
                chem_list.append(el[0])
 
    return list(set(chem_list))

def save_dict_as_csv(dictionary, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write the header row, if needed
        # writer.writerow(['Element', 'Value'])
        for key, value in dictionary.items():
            writer.writerow([key, value])

def one_result(model, tokenizer, dataset, idx, set_length={'type': 'add', 'value': 50}, 
                  separator=None, source='test',device='cuda', **kwargs):
    """_summary_

    Args:
        model (_type_): _description_
        dataset (_type_): _description_
        idx (_type_): _description_
        tokenizer (_type_): _description_
        set_length (dict, optional): _description_. Defaults to {'type': 'add', 'value': 50}. Otherwise {'type': 'mul', 'value': 1.2}.
        separator (_type_, optional): _description_. Defaults to None.
        remove_header (bool, optional): _description_. Defaults to True.
        cut (_type_, optional): _description_. Defaults to None. Set ';' if you want cut off the output after ';'. 
        source (str, optional): _description_. Defaults to 'test'.
        device (str, optional): _description_. Defaults to 'cuda'.

    Returns:
        _type_: _description_
    """
    # get the text string from the dataset
    label = dataset[source][idx]['label']   # prompt
    text = dataset[source][idx]['text'] # true output text
    # remove separator in case it still exists in the end of the label
    label_wo_sep = label.replace(separator, '')
    label_wo_sep = label_wo_sep.split(separator_out)[0]
    origin = label_wo_sep.split(separator_out)[0].split(',')   #split by ',' and take the first part. This can limit the length of the input if there are multiple reactants.
    # sort the reactants by length, and take the median length of the reactants
    origin.sort(key=len)
    origin = origin[len(origin)//2]
    
    tok_origin = tokenizer(origin, padding=True, truncation=True, return_tensors="pt")  
    tok_origin = {key: tensor.to(device) for key, tensor in tok_origin.items()}
    tok_origin_len = tok_origin['input_ids'].shape[-1]
    
    tok_label = tokenizer(label, padding=True, truncation=True, return_tensors="pt")
    tok_label = {key: tensor.to(device) for key, tensor in tok_label.items()}
    tok_label_len = tok_label['input_ids'].shape[-1]
    
    if set_length['type']=='add':
        max_length = tok_label_len + int(set_length['value'])
    else: 
        max_length = tok_label_len + int(tok_origin_len*float(set_length['value']))   
    
    print('tok_origin_len, tok_label_len, max_length: ', tok_origin_len, tok_label_len, max_length)
    generated_ids = model.generate(**tok_label, max_length=max_length, **kwargs)
    # print('text: ', text)
    tok_text = tokenizer(text)
    t_text = tokenizer.decode(tok_text["input_ids"])
    p_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # replace unfamiliar characters with ''
    for bk in black_list:
        p_text = p_text.replace(bk, '')

    return {'p_text': p_text, 't_text': t_text, 'label': label}


# Utility function to evaluate models
def evaluate_models(model_dict, dataset, tokenizer, num_sample, header, 
                   data_source, gen_conf, adjust_gt_len, 
                   separator, set_length,device):

    columns = ['idx', 'mpid', 'label', 't_text', 't_eq']
    model_keys = list(model_dict.keys())
    for model_key in model_keys:
        columns.extend([f'p_text.{model_key}', f'p_eq.{model_key}'])
    
    append_row = 0
    df = pd.DataFrame(columns=columns)
    # random indices without replacement
    idx_list = sorted(random.sample(range(len(dataset[data_source])), num_sample))
    for h, idx in tqdm(enumerate(idx_list), desc="Processing"):
        row_dict = {}
        for model_key, model in model_dict.items():
            try:
                mpid = dataset[data_source][idx]['mpid']
                out_dict = one_result(model, tokenizer, dataset, idx, set_length=set_length, 
                            separator=separator, source=data_source ,device=device, **gen_conf) 
                label, t_text, p_text = out_dict['label'].strip(), out_dict['t_text'].strip(), out_dict['p_text'].strip()
                p_text = p_text.replace('\n', '')

                if adjust_gt_len is not None:
                    
                    p_text = p_text[:int(len(t_text)+adjust_gt_len)]

                t_eq, p_eq = t_text[len(label):].strip(), p_text[len(label):].strip()
                if len(p_eq) == 0:
                    p_text = p_text + 'Error'
                    p_eq = 'Error'
                
                # Calculate similarity
                row_dict.update({'idx': idx, 'mpid': mpid, 'label': label, 't_text': t_text, 't_eq': t_eq, f'p_text.{model_key}': p_text, f'p_eq.{model_key}': p_eq})
                append_row += 1

                print(f'[{idx}] t_text. : ', t_text) 
                print(f'[{idx}] p_text.{model_key}: ', p_text)
                print(f'[{idx}] t_eq. : ', t_eq) 
                print(f'[{idx}] p_eq.{model_key}: ', p_eq)

            except Exception as e:
                print(f"Error at idx={idx}: {e}")
                row_dict.update({f'p_text.{model_key}': p_text, f'p_eq.{model_key}': p_eq})
        
        if append_row > 0:
            df = df._append(row_dict, ignore_index=True)
            print(' - - - - - ')
        
        
        if h%20==0 or h==len(idx_list)-1:
            df.to_csv(join(save_dir, f'{header}_{data_source}_n{num_sample}.csv'))
            print(f'Saved {header}_{data_source}_n{num_sample}.csv at {h}th iteration')

    return df
