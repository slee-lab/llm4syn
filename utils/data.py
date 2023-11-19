import numpy as np 
import torch
from datasets import DatasetDict, Dataset


def process_dataset(data, index=None, te_ratio=0.1, separator=' || '):
    num = len(data)
    te_num = int(num*te_ratio)
    tr_num = num - te_num
    if index is None: 
        index = list(range(num))
    data_list = [
        {"label": data[i]['targets_string'][0] , "text": data[i]['reaction_string']} for i in index
        # Add more dictionaries here...
    ]
    # Convert the list of dictionaries to a dictionary of lists
    data_dict = {"label": [], "text": []}
    for data_point in data_list:
        data_dict["label"].append(data_point["label"])
        # data_dict["text"].append(data_point["text"])
        data_dict["text"].append(data_point["label"] + separator + data_point["text"])

    # Create a Dataset
    custom_dataset = Dataset.from_dict(data_dict)

    # Split the dataset into train and test
    train_dataset = custom_dataset.select(list(range(tr_num)))
    test_dataset = custom_dataset.select(list(range(tr_num, num)))

    # Create a DatasetDict with the desired format
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "test": test_dataset,
    })

    return dataset_dict
    

def dataset_lhs2rhs(data, index=None, te_ratio=0.1, separator=' == ', eqsplit='=='):
    num = len(data)
    te_num = int(num*te_ratio)
    tr_num = num - te_num
    if index is None: 
        index = list(range(num))
    data_list = [
        {"label": data[i]['targets_string'][0] , "text": data[i]['reaction_string']} for i in index
        # Add more dictionaries here...
    ]
    if separator is None:
        separator=''
    else:
        separator=str(separator)
    # Convert the list of dictionaries to a dictionary of lists
    data_dict = {"label": [], "text": []}
    for data_point in data_list:
        a = data_point["text"]
        lhs, rhs = a.split(eqsplit)
        data_dict["label"].append(lhs+separator)
        data_dict["text"].append(lhs+separator+rhs)

    # Create a Dataset
    custom_dataset = Dataset.from_dict(data_dict)

    # Split the dataset into train and test
    train_dataset = custom_dataset.select(list(range(tr_num)))
    test_dataset = custom_dataset.select(list(range(tr_num, num)))

    # Create a DatasetDict with the desired format
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "test": test_dataset,
    })

    return dataset_dict

def dataset_rhs2lhs(data, index=None, te_ratio=0.1, separator=' || ', eqsplit='=='):
    num = len(data)
    te_num = int(num*te_ratio)
    tr_num = num - te_num
    if index is None: 
        index = list(range(num))
    data_list = [
        {"label": data[i]['targets_string'][0] , "text": data[i]['reaction_string']} for i in index
        # Add more dictionaries here...
    ]
    if separator is None:
        separator=''
    else:
        separator=str(separator)
    # Convert the list of dictionaries to a dictionary of lists
    data_dict = {"label": [], "text": []}
    for data_point in data_list:
        a = data_point["text"]
        lhs, rhs = a.split(eqsplit)
        data_dict["label"].append(rhs)
        data_dict["text"].append(rhs+separator+lhs)

    # Create a Dataset
    custom_dataset = Dataset.from_dict(data_dict)

    # Split the dataset into train and test
    train_dataset = custom_dataset.select(list(range(tr_num)))
    test_dataset = custom_dataset.select(list(range(tr_num, num)))

    # Create a DatasetDict with the desired format
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "test": test_dataset,
    })

    return dataset_dict


def dataset_ope2ceq(data, index=None, te_ratio=0.1, separator=' || '):
    num = len(data)
    te_num = int(num*te_ratio)
    tr_num = num - te_num
    if index is None: 
        index = list(range(num))
    data_list = [
        {
            "target": ", ".join(data[i]['targets_string']) if isinstance(data[i]['targets_string'], list) else data[i]['targets_string'],
            "opes": data[i]['operations'],
            'eq': data[i]['reaction_string']
        }
        for i in index
        # Add more dictionaries here...
    ]
    if separator is None:
        separator=''
    else:
        separator=str(separator)
    # Convert the list of dictionaries to a dictionary of lists
    data_dict = {"label": [], "text": []}
    for d in data_list:
        prompt = d['target'] + ': ' + str(d['opes'])
        data_dict["label"].append(prompt)
        data_dict["text"].append(prompt+ separator + d['eq'])

    # Create a Dataset
    custom_dataset = Dataset.from_dict(data_dict)

    # Split the dataset into train and test
    train_dataset = custom_dataset.select(list(range(tr_num)))
    test_dataset = custom_dataset.select(list(range(tr_num, num)))

    # Create a DatasetDict with the desired format
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "test": test_dataset,
    })

    return dataset_dict

def dataset_ceq2ope(data, index=None, te_ratio=0.1, separator=' || '):
    num = len(data)
    te_num = int(num*te_ratio)
    tr_num = num - te_num
    if index is None: 
        index = list(range(num))
    data_list = [
        {
            "target": ", ".join(data[i]['targets_string']) if isinstance(data[i]['targets_string'], list) else data[i]['targets_string'],
            "opes": data[i]['operations'],
            'eq': data[i]['reaction_string']
        }
        for i in index
        # Add more dictionaries here...
    ]
    if separator is None:
        separator=''
    else:
        separator=str(separator)
    # Convert the list of dictionaries to a dictionary of lists
    data_dict = {"label": [], "text": []}
    for d in data_list:
        prompt = d['eq']
        protocol = [d_['type'] for d_ in d['opes']] 
        data_dict["label"].append(prompt)
        data_dict["text"].append(prompt+ separator +" ".join(protocol))

    # Create a Dataset
    custom_dataset = Dataset.from_dict(data_dict)

    # Split the dataset into train and test
    train_dataset = custom_dataset.select(list(range(tr_num)))
    test_dataset = custom_dataset.select(list(range(tr_num, num)))

    # Create a DatasetDict with the desired format
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "test": test_dataset,
    })

    return dataset_dict


def dataset_ope2ceq_2(data, index=None, te_ratio=0.1, separator=' || '):
    num = len(data)
    te_num = int(num*te_ratio)
    tr_num = num - te_num
    if index is None: 
        index = list(range(num))
    data_list = [
        {
            "target": ", ".join(data[i]['targets_string']) if isinstance(data[i]['targets_string'], list) else data[i]['targets_string'],
            "opes": data[i]['operations'],
            'eq': data[i]['reaction_string']
        }
        for i in index
        # Add more dictionaries here...
    ]
    if separator is None:
        separator=''
    else:
        separator=str(separator)
    # Convert the list of dictionaries to a dictionary of lists
    data_dict = {"label": [], "text": []}
    for h, d in enumerate(data_list):
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
        data_dict["label"].append(prompt)
        data_dict["text"].append(prompt+ separator + d['eq'])

    # Create a Dataset
    custom_dataset = Dataset.from_dict(data_dict)

    # Split the dataset into train and test
    train_dataset = custom_dataset.select(list(range(tr_num)))
    test_dataset = custom_dataset.select(list(range(tr_num, num)))

    # Create a DatasetDict with the desired format
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "test": test_dataset,
    })

    return dataset_dict


def dataset_ceq2ope_2(data, index=None, te_ratio=0.1, separator=' || '):
    num = len(data)
    te_num = int(num*te_ratio)
    tr_num = num - te_num
    if index is None: 
        index = list(range(num))
    data_list = [
        {
            "target": ", ".join(data[i]['targets_string']) if isinstance(data[i]['targets_string'], list) else data[i]['targets_string'],
            "opes": data[i]['operations'],
            'eq': data[i]['reaction_string']
        }
        for i in index
        # Add more dictionaries here...
    ]
    if separator is None:
        separator=''
    else:
        separator=str(separator)
    # Convert the list of dictionaries to a dictionary of lists
    data_dict = {"label": [], "text": []}
    for h, d in enumerate(data_list):
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
        data_dict["label"].append(prompt)
        data_dict["text"].append(prompt+ separator +str(operations))

    # Create a Dataset
    custom_dataset = Dataset.from_dict(data_dict)

    # Split the dataset into train and test
    train_dataset = custom_dataset.select(list(range(tr_num)))
    test_dataset = custom_dataset.select(list(range(tr_num, num)))

    # Create a DatasetDict with the desired format
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "test": test_dataset,
    })

    return dataset_dict


def dataset_ope2ceq_3(data, index=None, te_ratio=0.1, separator=' || '):
    num = len(data)
    te_num = int(num*te_ratio)
    tr_num = num - te_num
    if index is None: 
        index = list(range(num))
    # print('index: ', index)
    data_list = [
        {
            "target": ", ".join(data[i]['targets_string']) if isinstance(data[i]['targets_string'], list) else data[i]['targets_string'],
            "opes": data[i]['operations'],
            'eq': data[i]['reaction_string']
        }
        for i in index
        # Add more dictionaries here...
    ]
    # print('data_list: ', data_list)
    if separator is None:
        separator=''
    else:
        separator=str(separator)
    # Convert the list of dictionaries to a dictionary of lists
    data_dict = {"label": [], "text": []}
    for h, d in enumerate(data_list):
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
        data_dict["label"].append(prompt)
        data_dict["text"].append(prompt+ separator + d['eq'])

    # Create a Dataset
    custom_dataset = Dataset.from_dict(data_dict)

    # Split the dataset into train and test
    train_dataset = custom_dataset.select(list(range(tr_num)))
    test_dataset = custom_dataset.select(list(range(tr_num, num)))

    # Create a DatasetDict with the desired format
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "test": test_dataset,
    })

    return dataset_dict


def dataset_ceq2ope_3(data, index=None, te_ratio=0.1, separator=' || '):
    num = len(data)
    te_num = int(num*te_ratio)
    tr_num = num - te_num
    if index is None: 
        index = list(range(num))
    data_list = [
        {
            "target": ", ".join(data[i]['targets_string']) if isinstance(data[i]['targets_string'], list) else data[i]['targets_string'],
            "opes": data[i]['operations'],
            'eq': data[i]['reaction_string']
        }
        for i in index
        # Add more dictionaries here...
    ]
    if separator is None:
        separator=''
    else:
        separator=str(separator)
    # Convert the list of dictionaries to a dictionary of lists
    data_dict = {"label": [], "text": []}
    for h, d in enumerate(data_list):
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
        data_dict["label"].append(prompt)
        data_dict["text"].append(prompt+ separator +str(operations))

    # Create a Dataset
    custom_dataset = Dataset.from_dict(data_dict)

    # Split the dataset into train and test
    train_dataset = custom_dataset.select(list(range(tr_num)))
    test_dataset = custom_dataset.select(list(range(tr_num, num)))

    # Create a DatasetDict with the desired format
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "test": test_dataset,
    })

    return dataset_dict



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
    if separator is not None:
        separator=str(separator)
        sep_inputs = tokenizer(separator, padding=True, truncation=True, return_tensors="pt")
        sep_inputs = {key: tensor.to(device) for key, tensor in sep_inputs.items()}
        len_sep = sep_inputs['input_ids'].shape[-1]   #len(separator)
    else: 
        separator=''
        len_sep = 0
    prompt = dataset[source][idx]['label'] + separator
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