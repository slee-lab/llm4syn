import random
from utils.data import *

class Dataset_LLM4SYN_Bench(LLMDataset):   # TODO: combine all the dataset class into one (or two)
    def __init__(self, data, index=None, te_ratio=0.1, separator='||', cut=None, arrow=arrow_l2r, task=None):
        super().__init__(data, index, te_ratio, separator, cut, arrow, task)

    def get_data_list(self):
        self.data_list = [
            {
                "target": ", ".join(self.data[i]['targets_string']) if isinstance(self.data[i]['targets_string'], list) else self.data[i]['targets_string'],
                "opes": self.data[i]['operations'],
                'eq': self.data[i]['reaction_string'].replace('==', self.arrow).split(self.cut)[0]
            }
            for i in self.index
        ]
        
    def get_data_dict(self):
        self.data_dict = {"label": [], "text": []}
        for h, d in enumerate(self.data_list):
            nopes = random.randint(1, 19)
            protocol = random.choices(opes_list, k=nopes)
            tgt = d['target']
            eq = d['eq']
            lhs,rhs = eq.split(self.arrow)
            # catalog = {'task': self.task, 'separator': self.separator, 'eq': eq, 'lhs': lhs, 'rhs': rhs, 'tgt': tgt, 'ope': '['+" ".join(protocol)+']'}
            catalog = {'task': self.task, 'separator': self.separator, 'eq': eq, 'lhs': lhs, 'rhs': rhs, 'tgt': tgt, 'ope': " ".join(protocol)}
            label, text = label_text(**catalog)
            self.data_dict["label"].append(label)
            self.data_dict["text"].append(text)