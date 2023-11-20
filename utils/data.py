import numpy as np 
import torch
from datasets import DatasetDict, Dataset


class LLMDataset:
    def __init__(self, data, index=None, te_ratio=0.1, separator=' || '):
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
    def __init__(self, data, index=None, te_ratio=0.1, separator=' || '):
        super().__init__(data, index, te_ratio, separator)

    def get_data_list(self):
        self.data_list = [
        ]
        
    def get_data_dict(self):
        self.data_dict = {"label": [], "text": []}


class Dataset_Lhs2Rhs(LLMDataset):
    def __init__(self, data, index=None, te_ratio=0.1, separator=' || ', eqsplit='=='):
        super().__init__(data, index, te_ratio, separator)
        self.eqsplit = eqsplit

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
            self.data_dict["label"].append(lhs+self.separator)
            self.data_dict["text"].append(lhs+self.separator+rhs)
            

class Dataset_Rhs2Lhs(LLMDataset):
    def __init__(self, data, index=None, te_ratio=0.1, separator=' || ', eqsplit='=='):
        super().__init__(data, index, te_ratio, separator)
        self.eqsplit = eqsplit

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
            self.data_dict["label"].append(rhs + self.separator)    #!
            self.data_dict["text"].append(rhs+self.separator+lhs)


class Dataset_Ope2Ceq(LLMDataset):
    def __init__(self, data, index=None, te_ratio=0.1, separator=' || '):
        super().__init__(data, index, te_ratio, separator)

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
            prompt = d['target'] + ': ' + str(d['opes'])
            self.data_dict["label"].append(prompt+ self.separator)  #!
            self.data_dict["text"].append(prompt+ self.separator + d['eq'])

class Dataset_Ceq2Ope(LLMDataset):
    def __init__(self, data, index=None, te_ratio=0.1, separator=' || '):
        super().__init__(data, index, te_ratio, separator)

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
            prompt = d['eq']
            protocol = [d_['type'] for d_ in d['opes']] 
            self.data_dict["label"].append(prompt+ self.separator)  #!
            self.data_dict["text"].append(prompt+ self.separator +" ".join(protocol))




class Dataset_Ope2Ceq_2(LLMDataset):
    def __init__(self, data, index=None, te_ratio=0.1, separator=' || '):
        super().__init__(data, index, te_ratio, separator)

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
            prompt = d['target'] + ': ' + str(operations)
            self.data_dict["label"].append(prompt + self.separator) #!
            self.data_dict["text"].append(prompt + self.separator + d['eq'])

class Dataset_Ceq2Ope_2(LLMDataset):
    def __init__(self, data, index=None, te_ratio=0.1, separator=' || '):
        super().__init__(data, index, te_ratio, separator)

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
            prompt = d['eq']
            self.data_dict["label"].append(prompt + self.separator) #!
            self.data_dict["text"].append(prompt+ self.separator +str(operations))


class Dataset_Ope2Ceq_3(LLMDataset):
    def __init__(self, data, index=None, te_ratio=0.1, separator=' || '):
        super().__init__(data, index, te_ratio, separator)

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
            prompt = d['target'] + ': ' + str(operations)
            self.data_dict["label"].append(prompt+ self.separator)  #!
            self.data_dict["text"].append(prompt+ self.separator + d['eq'])



class Dataset_Ceq2Ope_3(LLMDataset):
    def __init__(self, data, index=None, te_ratio=0.1, separator=' || '):
        super().__init__(data, index, te_ratio, separator)

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
            prompt = d['eq']
            self.data_dict["label"].append(prompt + self.separator) #!
            self.data_dict["text"].append(prompt+ self.separator +str(operations))


class Dataset_Tgt2Ceq(LLMDataset):
    def __init__(self, data, index=None, te_ratio=0.1, separator=' || '):
        super().__init__(data, index, te_ratio, separator)

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
            self.data_dict["label"].append(prompt+ self.separator)  #!
            self.data_dict["text"].append(prompt+ self.separator + d['eq'])
 



def show_one_test(model, dataset, idx, tokenizer, set_length={'type': 'add', 'value': 50}, separator=None, remove_header=True, source='test',device='cuda'):
    """_summary_

    Args:
        model (_type_): _description_
        dataset (_type_): _description_
        idx (_type_): _description_
        tokenizer (_type_): _description_
        set_length (dict, optional): _description_. Defaults to {'type': 'add', 'value': 50} or {'type': 'mul', 'value': 50}.
        show_header (bool, optional): _description_. Defaults to False.
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
    print('text: ', text)   #!
    encoded_text = tokenizer(text)
    if remove_header:
        # gt_answer = tokenizer.decode(encoded_text["input_ids"][input_length+len_sep:])
        # output = tokenizer.batch_decode(generated_ids[:, input_length+len_sep:], skip_special_tokens=True)[0]
        gt_answer = tokenizer.decode(encoded_text["input_ids"][max(input_length-1, 0):])
        output = tokenizer.batch_decode(generated_ids[:, max(input_length-1, 0):], skip_special_tokens=True)[0]
    else: 
        gt_answer = tokenizer.decode(encoded_text["input_ids"])
        output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print('gtruth: ', gt_answer)   #!
    print('answer: ', output)    #!
    return {'answer': output, 'text': text, 'label': prompt}


def preprocess_function(tokenizer, function, examples):
    return tokenizer([function(x) for x in examples["label"]])


def input_prcess(data):
    pass


def output_process(data):
    pass