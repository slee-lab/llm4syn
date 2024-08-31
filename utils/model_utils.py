import numpy as np 
import torch
from transformers import Trainer, AutoModelForCausalLM, TrainingArguments, AutoTokenizer
import torch.nn as nn


allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789()[]<>+-*=/., "

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

def setup_tokenizer(tk_model, pad_tokenizer):
    tokenizer = AutoTokenizer.from_pretrained(tk_model)
    if pad_tokenizer:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_model(model_name, hf_model, load_pretrained, device):
    if load_pretrained:
        return AutoModelForCausalLM.from_pretrained(model_name).to(device)
    else:
        return AutoModelForCausalLM.from_pretrained(hf_model).to(device)

class CustomLogitsProcessor:
    def __init__(self, allowed_tokens):
        self.allowed_tokens = allowed_tokens

    def __call__(self, input_ids, scores):
        # Mask logits for all tokens not in the allowed set
        mask = torch.ones_like(scores) * -float('inf')
        mask[:, self.allowed_tokens] = 0
        return scores + mask



class CustomTrainer_CE(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        labels = inputs.get("labels")

        # Compute cross-entropy loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, model.config.vocab_size), labels.view(-1))
        # loss = loss_fct(logits.squeeze(), labels.squeeze())

        return (loss, outputs) if return_outputs else loss
    

class CustomTrainer(Trainer):
    def __init__(self, *args, use_label_smoothing=True, epsilon=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_label_smoothing = use_label_smoothing
        self.epsilon = epsilon
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.use_label_smoothing and self.training:
            # Use Label Smoothing during training
            loss = self.label_smoothing_loss(logits, labels)
        else:
            # Use standard Cross-Entropy during evaluation
            loss = self.cross_entropy_loss(logits.view(-1, logits.size(-1)), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

    def label_smoothing_loss(self, logits, labels):
        log_probs = -nn.functional.log_softmax(logits, dim=-1)
        labels = labels.unsqueeze(-1)
        padding_mask = labels.eq(-100)  # Assuming -100 is the ignore_index for padding
        labels = torch.clamp(labels, min=0)
        nll_loss = log_probs.gather(dim=-1, index=labels)
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True)

        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)

        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        nll_loss = nll_loss.sum() / num_active_elements
        smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])

        return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss

    def evaluate(self, eval_dataset=None):
        self.use_label_smoothing = False  # Disable label smoothing during evaluation
        eval_output = super().evaluate(eval_dataset)
        self.use_label_smoothing = True   # Re-enable label smoothing for training
        return eval_output


if __name__ == "__main__":
    # Example usage
    train_dataset = ...  # Define this with your dataset
    eval_dataset = ...   # Define this with your dataset
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    training_args = TrainingArguments(output_dir="output_dir", evaluation_strategy="epoch")
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,  # Define this with your dataset
        eval_dataset=eval_dataset,    # Define this with your dataset
        use_label_smoothing=True,     # Enable label smoothing during training
        epsilon=0.1                   # Smoothing factor
    )

    # Start training
    trainer.train()

    # Evaluate and calculate perplexity using Cross-Entropy loss
    eval_results = trainer.evaluate()
    eval_loss = eval_results['eval_loss']
    perplexity = torch.exp(torch.tensor(eval_loss))
    print(f"Perplexity: {perplexity.item()}")


