import numpy as np 
import torch
from datasets import DatasetDict, Dataset
import inspect

def make_dict(variables):
    frame = inspect.currentframe().f_back
    dictionary = {name: value for name, value in frame.f_locals.items() if id(value) in [id(var) for var in variables]}
    return dictionary

eqsplit = '=='
remove_ing_exception = {'Shaping': 'Shape'}

def space_separator(target_string: str, separator: str) -> str:
    # Find the index of the separator in the target string
    separator = str(separator).strip()
    index = target_string.find(separator)
    
    while index != -1:
        # Check if there is no space before the separator
        if index > 0 and target_string[index - 1] != ' ':
            target_string = target_string[:index] + ' ' + target_string[index:]
            index += 1  # Adjust index to account for the inserted space
        
        # Check if there is no space after the separator
        index += len(separator)
        if index < len(target_string) and target_string[index] != ' ':
            target_string = target_string[:index] + ' ' + target_string[index:]
        
        # Find the next occurrence of the separator
        index = target_string.find(separator, index)
    
    return target_string

class LLMDataset:
    def __init__(self, data, index=None, te_ratio=0.1, separator=' || ', cut=None, task=None):
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
        self.task = task
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


class Dataset_template(LLMDataset):
    def __init__(self, data, index=None, te_ratio=0.1, separator='||', cut=None, task=None):
        super().__init__(data, index, te_ratio, separator, cut, task)

    def get_data_list(self):
        self.data_list = [
        ]
        
    def get_data_dict(self):
        self.data_dict = {"label": [], "text": []}


           

# class Dataset_Ceq2Ope_simple(LLMDataset):  #TODO: remove in the end
#     def __init__(self, data, index=None, te_ratio=0.1, separator='||', cut=None, task=None):
#         super().__init__(data, index, te_ratio, separator, cut, task)

#     def get_data_list(self):
#         self.data_list = [
#             {
#                 "target": ", ".join(self.data[i]['targets_string']) if isinstance(self.data[i]['targets_string'], list) else self.data[i]['targets_string'],
#                 "opes": self.data[i]['operations'],
#                 'eq': self.data[i]['reaction_string'].replace('==', '->')
#             }
#             for i in self.index
#             # Add more dictionaries here...
#         ]
        
#     def get_data_dict(self):
#         self.data_dict = {"label": [], "text": []}
#         for d in self.data_list:
#             eq = d['eq']
#             if self.cut in eq:
#                 eq = eq.split(self.cut)[0]
#             prompt = eq
#             protocol = [d_['type'] for d_ in d['opes']] 
#             protocol_ = [x.replace('Operation', '') for x in protocol]   #!
#             protocol = []
#             for x in protocol_:
#                 if x in remove_ing_exception.keys():
#                     protocol.append(x.replace(x, remove_ing_exception[x]))
#                 else: 
#                     protocol.append(x)
#             protocol = [x.replace('ing', '') for x in protocol]   #!
#             self.data_dict["label"].append(space_separator(prompt+ self.separator, self.separator))  #!
#             self.data_dict["text"].append(space_separator(prompt+ self.separator +" ".join(protocol), self.separator))

            
arrow_l2r = '->'
separator_o = ': '

def label_text(task='lhs2rhs', separator_o=separator_o, lhs=None, rhs=None, tgt=None, eq=None, ope=None, separator='->'):
    if task == 'lhs2rhs':
        label = space_separator(lhs + separator, separator)
        text = space_separator(lhs + separator + rhs, separator)
    elif task == 'rhs2lhs':
        label = space_separator(rhs + separator, separator)
        text = space_separator(rhs + separator + lhs, separator)
    elif task == 'tgt2ceq':
        label = space_separator(tgt + separator, separator)
        text = space_separator(tgt + separator + eq, separator)
    elif task == 'lhsope2rhs':
        label = space_separator(lhs + separator_o + ope + separator, separator)
        text = space_separator(lhs + separator_o + ope + separator + rhs, separator)
    elif task == 'rhsope2lhs':
        label = space_separator(rhs + separator_o + ope + separator, separator)
        text = space_separator(rhs + separator_o + ope + separator + lhs, separator)
    elif task == 'tgtope2ceq':
        label = space_separator(tgt + separator_o + ope + separator, separator)
        text = space_separator(tgt + separator_o + ope + separator + eq, separator)
    
    return label, text
        


class Dataset_LLM4SYN(LLMDataset):   # TODO: combine all the dataset class into one (or two)
    def __init__(self, data, index=None, te_ratio=0.1, separator='||', cut=None, task=None):
        super().__init__(data, index, te_ratio, separator, cut, task)

    def get_data_list(self):
        self.data_list = [
            {
                "target": ", ".join(self.data[i]['targets_string']) if isinstance(self.data[i]['targets_string'], list) else self.data[i]['targets_string'],
                "opes": self.data[i]['operations'],
                'eq': self.data[i]['reaction_string'].replace('==', arrow_l2r).split(self.cut)[0]
            }
            for i in self.index
        ]
        
    def get_data_dict(self):
        self.data_dict = {"label": [], "text": []}
        for h, d in enumerate(self.data_list):
            protocol_ = [d_['type'] for d_ in d['opes']] 
            protocol_ = [x.replace('Operation', '') for x in protocol_]   #!
            protocol = []
            for x in protocol_:
                if x in remove_ing_exception.keys():
                    protocol.append(x.replace(x, remove_ing_exception[x]))
                else: 
                    protocol.append(x.replace('ing', ''))
            tgt = d['target']
            eq = d['eq']
            lhs,rhs = eq.split(arrow_l2r)
            catalog = {'task': self.task, 'separator': self.separator, 'eq': eq, 'lhs': lhs, 'rhs': rhs, 'tgt': tgt, 'ope': '['+" ".join(protocol)+']'}
            label, text = label_text(**catalog)
            self.data_dict["label"].append(label)
            self.data_dict["text"].append(text)



def one_result(model, tokenizer, dataset, idx, set_length={'type': 'add', 'value': 50}, 
                  separator=None, source='test',device='cuda'): 
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
    label = dataset[source][idx]['label']
    text = dataset[source][idx]['text']
    # remove separator if it is not None from label
    label_wo_sep = label.replace(separator, '')
    source = label_wo_sep.split(separator_o)[0]
    
    tok_source = tokenizer(source, padding=True, truncation=True, return_tensors="pt")
    tok_source = {key: tensor.to(device) for key, tensor in tok_source.items()}
    tok_source_len = tok_source['input_ids'].shape[-1]
    
    tok_label = tokenizer(label, padding=True, truncation=True, return_tensors="pt")
    tok_label = {key: tensor.to(device) for key, tensor in tok_label.items()}
    tok_label_len = tok_label['input_ids'].shape[-1]
    
    if set_length['type']=='add':
        max_length = tok_label_len + int(set_length['value'])
    else: 
        max_length = tok_label_len + int(tok_source_len*float(set_length['value']))   #TODO: multiply the length of input excluding the OPE part (split by ':' and take the first part)
    
    print('tok_source_len, tok_label_len, max_length: ', tok_source_len, tok_label_len, max_length)
    generated_ids = model.generate(**tok_label, max_length=max_length, repetition_penalty=1.1)
    # print('text: ', text)
    tok_text = tokenizer(text)
    gt_text = tokenizer.decode(tok_text["input_ids"])
    out_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # print('gtruth: ', gt_answer) 
    # print('answer: ', output)
    return {'out_text': out_text, 'gt_text': gt_text, 'label': label}




def preprocess_function(tokenizer, function, examples):
    return tokenizer([function(x) for x in examples["label"]])


def input_prcess(data):
    pass


def output_process(data):
    pass