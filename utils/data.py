import numpy as np 
import torch
from datasets import DatasetDict, Dataset


def process_dataset(data, index=None, te_ratio=0.1, connector=': '):
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
        data_dict["text"].append(data_point["label"] + connector + data_point["text"])

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
    

def equation_dataset(data, index=None, te_ratio=0.1):
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
        a = data_point["text"]
        data_dict["label"].append(a.split('==')[0] + '==')
        data_dict["text"].append(a)

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


def preprocess_function(tokenizer, function, examples):
    return tokenizer([function(x) for x in examples["label"]])


def input_prcess(data):
    pass


def output_process(data):
    pass