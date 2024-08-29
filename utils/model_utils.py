import numpy as np 
import torch
from transformers import Trainer
import torch.nn as nn

def get_epoch_lists(total_epochs, num_folds=5, ep_per_fold=1):
    # Initialize lists to store performance metrics for each fold
    epset = total_epochs//(num_folds*ep_per_fold)
    epset_frac = total_epochs%(num_folds*ep_per_fold)
    ep_lists = []
    for i in range(epset):
        ep_list = []
        for j in range(num_folds):
            ep_list.append(ep_per_fold)
        ep_lists.append(ep_list)
    total_sum = sum([sum(sublist) for sublist in ep_lists])
    if total_sum < total_epochs:
        ep_list = []
        while sum(ep_list)<epset_frac:
            ep_list.append(ep_per_fold)
        ep_lists.append(ep_list)

    print(ep_lists)
    total_sum = sum([sum(sublist) for sublist in ep_lists])
    print("Total epochs:", total_sum)
    return ep_lists


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        labels = inputs.get("labels")

        # Compute cross-entropy loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, model.config.vocab_size), labels.view(-1))

        return (loss, outputs) if return_outputs else loss