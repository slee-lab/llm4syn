import numpy as np 
import torch
from datasets import DatasetDict, Dataset


remove_ing_exception = {'Shaping': 'Shape'}

class LLMDataset:
    def __init__(self, data, index=None, te_ratio=0.1, separator=' || ', cut=None):
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
    def __init__(self, data, index=None, te_ratio=0.1, separator=' || ', cut=None):
        super().__init__(data, index, te_ratio, separator, cut)

    def get_data_list(self):
        self.data_list = [
        ]
        
    def get_data_dict(self):
        self.data_dict = {"label": [], "text": []}


class Dataset_Lhs2Rhs(LLMDataset):
    def __init__(self, data, index=None, te_ratio=0.1, separator=' || ', cut=None, eqsplit='=='):
        self.eqsplit = eqsplit
        super().__init__(data, index, te_ratio, separator, cut)
        # self.eqsplit = eqsplit

    def get_data_list(self):
        self.data_list = [
            {"label": self.data[i]['targets_string'][0] , "text": self.data[i]['reaction_string']} for i in self.index
            # Add more dictionaries here...
        ]
        
    def get_data_dict(self):
        self.data_dict = {"label": [], "text": []}
        for data_point in self.data_list:
            a = data_point["text"]
            lhs, rhs = a.split(self.eqsplit)
            if self.cut in rhs:
                rhs = rhs.split(self.cut)[0]
            self.data_dict["label"].append(lhs+self.separator)
            self.data_dict["text"].append(lhs+self.separator+rhs)
            

class Dataset_Rhs2Lhs(LLMDataset):
    def __init__(self, data, index=None, te_ratio=0.1, separator=' || ', cut=None, eqsplit='=='):
        self.eqsplit = eqsplit
        super().__init__(data, index, te_ratio, separator, cut)
        # self.eqsplit = eqsplit

    def get_data_list(self):
        self.data_list = [
            {"label": self.data[i]['targets_string'][0] , "text": self.data[i]['reaction_string']} for i in self.index
            # Add more dictionaries here...
        ]
    def get_data_dict(self):
        self.data_dict = {"label": [], "text": []}
        for data_point in self.data_list:
            a = data_point["text"]
            lhs, rhs = a.split(self.eqsplit)
            if self.cut in rhs:
                rhs = rhs.split(self.cut)[0]
            self.data_dict["label"].append(rhs + self.separator)    #!
            self.data_dict["text"].append(rhs+self.separator+lhs)


class Dataset_Ope2Ceq(LLMDataset):
    def __init__(self, data, index=None, te_ratio=0.1, separator=' || ', cut=None):
        super().__init__(data, index, te_ratio, separator, cut)

    def get_data_list(self):
        self.data_list = [
            {
                "target": ", ".join(self.data[i]['targets_string']) if isinstance(self.data[i]['targets_string'], list) else self.data[i]['targets_string'],
                "opes": self.data[i]['operations'],
                'eq': self.data[i]['reaction_string']
            }
            for i in self.index
        ]
        
    def get_data_dict(self):
        self.data_dict = {"label": [], "text": []}
        for d in self.data_list:
            eq = d['eq']
            if self.cut in eq:
                eq = eq.split(self.cut)[0]
            prompt = d['target'] + ': ' + str(d['opes'])
            self.data_dict["label"].append(prompt+ self.separator)  #!
            self.data_dict["text"].append(prompt+ self.separator + eq)

class Dataset_Ceq2Ope(LLMDataset):
    def __init__(self, data, index=None, te_ratio=0.1, separator=' || ', cut=None):
        super().__init__(data, index, te_ratio, separator, cut)

    def get_data_list(self):
        self.data_list = [
            {
                "target": ", ".join(self.data[i]['targets_string']) if isinstance(self.data[i]['targets_string'], list) else self.data[i]['targets_string'],
                "opes": self.data[i]['operations'],
                'eq': self.data[i]['reaction_string']
            }
            for i in self.index
            # Add more dictionaries here...
        ]
        
    def get_data_dict(self):
        self.data_dict = {"label": [], "text": []}
        for d in self.data_list:
            eq = d['eq']
            if self.cut in eq:
                eq = eq.split(self.cut)[0]
            prompt = eq
            protocol = [d_['type'] for d_ in d['opes']] 
            protocol_ = [x.replace('Operation', '') for x in protocol]   #!
            self.data_dict["label"].append(prompt+ self.separator)  #!
            self.data_dict["text"].append(prompt+ self.separator +" ".join(protocol))




class Dataset_Ope2Ceq_2(LLMDataset):
    def __init__(self, data, index=None, te_ratio=0.1, separator=' || ', cut=None):
        super().__init__(data, index, te_ratio, separator, cut)

    def get_data_list(self):
        self.data_list = [
            {
                "target": ", ".join(self.data[i]['targets_string']) if isinstance(self.data[i]['targets_string'], list) else self.data[i]['targets_string'],
                "opes": self.data[i]['operations'],
                'eq': self.data[i]['reaction_string']
            }
            for i in self.index
        ]
        
    def get_data_dict(self):
        self.data_dict = {"label": [], "text": []}
        for h, d in enumerate(self.data_list):
            operations = {}
            for i, d_ in enumerate(d['opes']):
                ope = d_['type']
                if d_['conditions'] is None:
                    op = None
                    # print([h, i], ope, op)
                else:
                    op = {k:a for k, a in d_['conditions'].items() if all([len(a)>0, a is not None])}
                operations[ope]=op
                # print([h, i], ope, op)
            eq = d['eq']
            if self.cut in eq:
                eq = eq.split(self.cut)[0]
            prompt = d['target'] + ': ' + str(operations)
            self.data_dict["label"].append(prompt + self.separator) #!
            self.data_dict["text"].append(prompt + self.separator + eq)

class Dataset_Ceq2Ope_2(LLMDataset):
    def __init__(self, data, index=None, te_ratio=0.1, separator=' || ', cut=None):
        super().__init__(data, index, te_ratio, separator, cut)

    def get_data_list(self):
        self.data_list = [
            {
                "target": ", ".join(self.data[i]['targets_string']) if isinstance(self.data[i]['targets_string'], list) else self.data[i]['targets_string'],
                "opes": self.data[i]['operations'],
                'eq': self.data[i]['reaction_string']
            }
            for i in self.index
        ]
        
    def get_data_dict(self):
        self.data_dict = {"label": [], "text": []}
        for h, d in enumerate(self.data_list):
            operations = {}
            for i, d_ in enumerate(d['opes']):
                ope = d_['type']
                if d_['conditions'] is None:
                    op = None
                    # print([h, i], ope, op)
                else:
                    op = {k:a for k, a in d_['conditions'].items() if all([len(a)>0, a is not None])}
                    # print([h, i], ope, op)
                operations[ope]=op
            eq = d['eq']
            if self.cut in eq:
                eq = eq.split(self.cut)[0]
            prompt = eq
            self.data_dict["label"].append(prompt + self.separator) #!
            self.data_dict["text"].append(prompt+ self.separator +str(operations))


class Dataset_Ope2Ceq_3(LLMDataset):
    def __init__(self, data, index=None, te_ratio=0.1, separator=' || ', cut=None):
        super().__init__(data, index, te_ratio, separator, cut)

    def get_data_list(self):
        self.data_list = [
            {
                "target": ", ".join(self.data[i]['targets_string']) if isinstance(self.data[i]['targets_string'], list) else self.data[i]['targets_string'],
                "opes": self.data[i]['operations'],
                'eq': self.data[i]['reaction_string']
            }
            for i in self.index
        ]
        
    def get_data_dict(self):
        self.data_dict = {"label": [], "text": []}
        for h, d in enumerate(self.data_list):
            operations = {}
            for i, d_ in enumerate(d['opes']):
                ope = d_['type']
                if d_['conditions'] is None:
                    op = None
                    # print([h, i], ope, op)
                else:
                    op = {}
                    for k, a in d_['conditions'].items():
                        if a is not None:
                            if len(a) > 0:
                                op[k] = a
                operations[ope]=op
                # print([h, i], ope, op)
            eq = d['eq']
            if self.cut in eq:
                eq = eq.split(self.cut)[0]
            prompt = d['target'] + ': ' + str(operations)
            self.data_dict["label"].append(prompt+ self.separator)  #!
            self.data_dict["text"].append(prompt+ self.separator + eq)



class Dataset_Ceq2Ope_3(LLMDataset):
    def __init__(self, data, index=None, te_ratio=0.1, separator=' || ', cut=None):
        super().__init__(data, index, te_ratio, separator, cut)

    def get_data_list(self):
        self.data_list = [
            {
                "target": ", ".join(self.data[i]['targets_string']) if isinstance(self.data[i]['targets_string'], list) else self.data[i]['targets_string'],
                "opes": self.data[i]['operations'],
                'eq': self.data[i]['reaction_string']
            }
            for i in self.index
        ]
        
    def get_data_dict(self):
        self.data_dict = {"label": [], "text": []}
        for h, d in enumerate(self.data_list):
            operations = {}
            for i, d_ in enumerate(d['opes']):
                ope = d_['type']
                # print(ope, d_['conditions'])
                if d_['conditions'] is None:
                    op = None
                    # print([h, i], ope, op)
                else:
                    op = {}
                    for k, a in d_['conditions'].items():
                        if a is not None:
                            if len(a) > 0:
                                op[k] = a
                operations[ope]=op
                # print([h, i], ope, op)
            eq = d['eq']
            if self.cut in eq:
                eq = eq.split(self.cut)[0]
            prompt = eq
            self.data_dict["label"].append(prompt + self.separator) #!
            self.data_dict["text"].append(prompt+ self.separator +str(operations))


class Dataset_Ceq2Ope_4(LLMDataset):
    def __init__(self, data, index=None, te_ratio=0.1, separator=' || ', cut=None):
        super().__init__(data, index, te_ratio, separator, cut)

    def get_data_list(self):
        self.data_list = [
            {
                "target": ", ".join(self.data[i]['targets_string']) if isinstance(self.data[i]['targets_string'], list) else self.data[i]['targets_string'],
                "opes": self.data[i]['operations'],
                'eq': self.data[i]['reaction_string']
            }
            for i in self.index
        ]
        
    def get_data_dict(self):
        self.data_dict = {"label": [], "text": []}
        for h, d in enumerate(self.data_list):
            operations = {}
            for i, d_ in enumerate(d['opes']):
                ope = d_['type']
                # print(ope, d_['conditions'])
                if d_['conditions'] is None:
                    op = None
                    # print([h, i], ope, op)
                else:
                    op = {}
                    for k, a in d_['conditions'].items():
                        if a is not None:
                            if len(a) > 0:
                                op[k] = a
                operations[ope]=op
                # print([h, i], ope, op)
            eq = d['eq']
            if self.cut in eq:
                eq = eq.split(self.cut)[0]
            prompt = eq
            self.data_dict["label"].append(prompt + self.separator) #!
            self.data_dict["text"].append(prompt+ self.separator +str(operations))


class Dataset_Ope2Ceq_simple(LLMDataset):
    def __init__(self, data, index=None, te_ratio=0.1, separator=' || ', cut=None):
        super().__init__(data, index, te_ratio, separator, cut)

    def get_data_list(self):
        self.data_list = [
            {
                "target": ", ".join(self.data[i]['targets_string']) if isinstance(self.data[i]['targets_string'], list) else self.data[i]['targets_string'],
                "opes": self.data[i]['operations'],
                'eq': self.data[i]['reaction_string']
            }
            for i in self.index
        ]
        
    def get_data_dict(self):
        self.data_dict = {"label": [], "text": []}
        for h, d in enumerate(self.data_list):
            protocol = [d_['type'] for d_ in d['opes']] 
            protocol_ = [x.replace('Operation', '') for x in protocol]   #!
            protocol = []
            for x in protocol_:
                if x in remove_ing_exception.keys():
                    protocol.append(x.replace(x, remove_ing_exception[x]))
                else: 
                    protocol.append(x)
            protocol = [x.replace('ing', '') for x in protocol]   #!
                # print([h, i], ope, op)
            eq = d['eq']
            if self.cut in eq:
                eq = eq.split(self.cut)[0]
            prompt = d['target'] + ': ' +  " ".join(protocol)
            self.data_dict["label"].append(prompt+ self.separator)  #!
            self.data_dict["text"].append(prompt+ self.separator + eq)
            

class Dataset_Ceq2Ope_simple(LLMDataset):
    def __init__(self, data, index=None, te_ratio=0.1, separator=' || ', cut=None):
        super().__init__(data, index, te_ratio, separator, cut)

    def get_data_list(self):
        self.data_list = [
            {
                "target": ", ".join(self.data[i]['targets_string']) if isinstance(self.data[i]['targets_string'], list) else self.data[i]['targets_string'],
                "opes": self.data[i]['operations'],
                'eq': self.data[i]['reaction_string']
            }
            for i in self.index
            # Add more dictionaries here...
        ]
        
    def get_data_dict(self):
        self.data_dict = {"label": [], "text": []}
        for d in self.data_list:
            eq = d['eq']
            if self.cut in eq:
                eq = eq.split(self.cut)[0]
            prompt = eq
            protocol = [d_['type'] for d_ in d['opes']] 
            protocol_ = [x.replace('Operation', '') for x in protocol]   #!
            protocol = []
            for x in protocol_:
                if x in remove_ing_exception.keys():
                    protocol.append(x.replace(x, remove_ing_exception[x]))
                else: 
                    protocol.append(x)
            protocol = [x.replace('ing', '') for x in protocol]   #!
            self.data_dict["label"].append(prompt+ self.separator)  #!
            self.data_dict["text"].append(prompt+ self.separator +" ".join(protocol))



class Dataset_Tgt2Ceq(LLMDataset):
    def __init__(self, data, index=None, te_ratio=0.1, separator=' || ', cut=None):
        super().__init__(data, index, te_ratio, separator, cut)

    def get_data_list(self):
        self.data_list = [
            {
                "target": ", ".join(self.data[i]['targets_string']) if isinstance(self.data[i]['targets_string'], list) else self.data[i]['targets_string'],
                'eq': self.data[i]['reaction_string']
            }
            for i in self.index
        ]
        
    def get_data_dict(self):
        self.data_dict = {"label": [], "text": []}
        for h, d in enumerate(self.data_list):
            prompt = d['target']
            eq = d['eq']
            if self.cut in eq:
                eq = eq.split(self.cut)[0]
            self.data_dict["label"].append(prompt+ self.separator)  #!
            self.data_dict["text"].append(prompt+ self.separator + eq)
 

class Dataset_Ceq(LLMDataset): # for BERT task
    def __init__(self, data, index=None, te_ratio=0.1, separator=' || ', cut=None):
        super().__init__(data, index, te_ratio, separator, cut)

    def get_data_list(self):
        self.data_list = [
            {
                # "target": ", ".join(self.data[i]['targets_string']) if isinstance(self.data[i]['targets_string'], list) else self.data[i]['targets_string'],
                'eq': self.data[i]['reaction_string']
            }
            for i in self.index
        ]
        
    def get_data_dict(self):
        self.data_dict = {"label": [], "text": []}
        # self.data_dict = {"text": []}
        for h, d in enumerate(self.data_list):
            # prompt = d['target']
            eq = d['eq']
            # if self.cut in eq:
            #     eq = eq.split(self.cut)[0]
            self.data_dict["label"].append(eq)  # no need
            self.data_dict["text"].append(eq)
 


def show_one_test(model, dataset, idx, tokenizer, set_length={'type': 'add', 'value': 50}, 
                  separator=None, remove_header=True, cut=None, source='test',device='cuda'):
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
    given = dataset[source][idx]['label']
    if separator is not None:
        separator=str(separator)
        sep_inputs = tokenizer(separator, padding=True, truncation=True, return_tensors="pt")
        sep_inputs = {key: tensor.to(device) for key, tensor in sep_inputs.items()}
        len_sep = sep_inputs['input_ids'].shape[-1]   #len(separator)
        if given.endswith(separator):
            prompt = given
        else: 
            prompt = given + separator
    else: 
        separator=''
        len_sep = 0
        prompt = given
    model_inputs = tokenizer(prompt, padding=True, truncation=True, return_tensors="pt")
    model_inputs = {key: tensor.to(device) for key, tensor in model_inputs.items()}
    input_length = model_inputs['input_ids'].shape[-1]
    if set_length['type']=='add':
        max_length = input_length + int(set_length['value'])
    else: 
        max_length = input_length*set_length['value']
    generated_ids = model.generate(**model_inputs, max_length=max_length)
    text = dataset[source][idx]['text']
    # print('text: ', text)
    encoded_text = tokenizer(text)
    if remove_header:
        gt_answer = tokenizer.decode(encoded_text["input_ids"][max(input_length-1, 0):])
        output = tokenizer.batch_decode(generated_ids[:, max(input_length-1, 0):], skip_special_tokens=True)[0]
        if cut is not None: #!
            gt_answer = gt_answer.split(cut)[0]
            output = output.split(cut)[0]
    else: 
        gt_answer = tokenizer.decode(encoded_text["input_ids"])
        output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        if cut is not None: 
            gt_answer_cut = gt_answer.split(cut)[0]
            if len(gt_answer_cut) >= len(prompt):
                gt_answer = gt_answer_cut
            output_cut = output.split(cut)[0]
            if len(output_cut) >= len(prompt):
                output = output_cut
    # print('gtruth: ', gt_answer) 
    # print('answer: ', output)
    return {'answer': output, 'text': text, 'label': prompt}


def preprocess_function(tokenizer, function, examples):
    return tokenizer([function(x) for x in examples["label"]])


def input_prcess(data):
    pass


def output_process(data):
    pass