import numpy as np 
import json
import torch
import random
from datasets import DatasetDict, Dataset
from utils.data_config import separator_out, arrow_l2r

eqsplit = '=='
remove_ing_exception = {'Shaping': 'Shape'}
opes_list = 'SolutionMix', 'Shape', 'Dry', 'Mix', 'Heat', 'LiquidGrind', 'Quench'


def format_separator(text: str, separator: str) -> str:
    # Step 0: Remove extra spaces
    while '  ' in text:
        text = text.replace('  ', ' ')
    # Step 1: Normalize the spaces around the separator
    # Use the str.replace to ensure one space on both sides of the separator
    formatted_text = text.replace(f' {separator}', separator).replace(f'{separator} ', separator)
    formatted_text = formatted_text.replace(separator, f' {separator} ')  # Add spaces around the separator
    
    # Step 2: Handle the case where the separator is at the end of the text
    if formatted_text.endswith(f' {separator} '):
        formatted_text = formatted_text[:-1]  # Remove trailing space after separator

    return formatted_text

class LLMDataset:
    def __init__(self, data, index=None, te_ratio=0.1, separator=' || ', cut=None, arrow=arrow_l2r, task=None):
        self.data = data
        self.num = len(self.data)
        self.te_ratio = te_ratio
        self.te_num = int(self.num*self.te_ratio)
        self.tr_num = self.num - self.te_num
        if index is None: 
            self.index = list(range(self.num))
        else: 
            self.index = index
        if separator is None:
            self.separator=''
        else:
            self.separator=str(separator)
        if cut is None:
            self.cut = '@@@@@@@@@@@@@@@'
        else: 
            self.cut = cut
        self.arrow = arrow
        self.task = task
        self.data_dict = None
        self.get_data_list()
        self.get_data_dict()
        self.get_dataset()
    
    def get_data_list(self):
        pass

    def get_data_dict(self):
        pass

    def get_dataset(self):
        custom_dataset = Dataset.from_dict(self.data_dict)

        # Split the dataset into train and test
        train_dataset = custom_dataset.select(list(range(self.tr_num)))
        test_dataset = custom_dataset.select(list(range(self.tr_num, self.num)))

        # Create a DatasetDict with the desired format
        self.dataset = DatasetDict({
            "train": train_dataset,
            "test": test_dataset,
        })


def label_text(task='lhs2rhs', separator_out=separator_out, lhs=None, rhs=None, tgt=None, eq=None, ope=None, separator='->'):
    if task == 'lhs2rhs':
        label = format_separator(lhs + separator, separator)
        text = format_separator(lhs + separator + rhs, separator)
    elif task == 'rhs2lhs':
        label = format_separator(rhs + separator, separator)
        text = format_separator(rhs + separator + lhs, separator)
    elif task == 'tgt2ceq':
        label = format_separator(tgt + separator, separator)
        text = format_separator(tgt + separator + eq, separator)
    elif task == 'lhsope2rhs':
        label = format_separator(lhs + separator_out + ope + separator, separator)
        text = format_separator(lhs + separator_out + ope + separator + rhs, separator)
    elif task == 'rhsope2lhs':
        label = format_separator(rhs + separator_out + ope + separator, separator)
        text = format_separator(rhs + separator_out + ope + separator + lhs, separator)
    elif task == 'tgtope2ceq':
        label = format_separator(tgt + separator_out + ope + separator, separator)
        text = format_separator(tgt + separator_out + ope + separator + eq, separator)
    return label, text
        

class Dataset_LLM4SYN(LLMDataset): 
    def __init__(self, data, index=None, te_ratio=0.1, separator='||', cut=None, arrow=arrow_l2r, task=None):
        super().__init__(data, index, te_ratio, separator, cut, arrow, task)

    def get_data_list(self):
        self.data_list = [
            {
                "target": ", ".join(self.data[i]['targets_string']) if isinstance(self.data[i]['targets_string'], list) else self.data[i]['targets_string'],
                "opes": self.data[i]['operations'],
                'eq': self.data[i]['reaction_string'].replace('==', self.arrow).split(self.cut)[0],
                'mpid': self.data[i]['target']['mp_id'] 
            }
            for i in self.index
        ]
        
    def get_data_dict(self):
        self.data_dict = {"label": [], "text": [], "mpid": []}
        for h, d in enumerate(self.data_list):
            protocol_ = [d_['type'] for d_ in d['opes']] 
            protocol_ = [x.replace('Operation', '') for x in protocol_]  
            protocol = []
            for x in protocol_:
                if x in remove_ing_exception.keys():
                    protocol.append(x.replace(x, remove_ing_exception[x]))
                else: 
                    protocol.append(x.replace('ing', ''))
            tgt = d['target']
            eq = d['eq']
            lhs,rhs = eq.split(self.arrow)
            catalog = {'task': self.task, 'separator': self.separator, 'eq': eq, 'lhs': lhs, 'rhs': rhs, 'tgt': tgt, 'ope': " ".join(protocol)}
            label, text = label_text(**catalog)
            self.data_dict["label"].append(label)
            self.data_dict["text"].append(text)
            self.data_dict["mpid"].append(d['mpid'])  


def load_and_sample_data(data_path, task, separator, te_ratio=0.1, cut=';', arrow='->', sample_ratio=1, save_idx_name=None):
    with open(data_path, 'r') as f:
        data = json.load(f)
    num_sample = int(len(data) * sample_ratio)
    rand_indices = random.sample(range(len(data)), num_sample)
    sampled_data = [data[i] for i in rand_indices]
    if save_idx_name is not None:
        with open(f'./models/{save_idx_name}/idx_s{sample_ratio}.txt', 'w') as f: 
            for i in rand_indices: f.write(f"{i}\n")
    return Dataset_LLM4SYN(sampled_data, index=None, te_ratio=te_ratio, separator=separator, cut=cut, arrow=arrow, task=task).dataset

