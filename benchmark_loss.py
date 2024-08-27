#%%
import os, sys
from os.path import join
from collections import defaultdict
from collections import Counter
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
import re
import numpy as np
import csv
from tqdm import tqdm
from itertools import product
from scipy.optimize import linear_sum_assignment
import re
import pandas as pd
from matplotlib import pyplot as plt

from utils.metrics import *

#%%
save_dir = '/home/rokabe/data2/llm4syn/save'
# csv_file = join(save_dir, 'ceq_simple_dgpt_v1.4_test_1948_v5.1_df.csv')
csv_file = join(save_dir, 'ope_simple_dgpt_v1.2_test_1948_v1.1_df.csv')
df = pd.read_csv(csv_file)
df['pred'] = df['pred'].apply(lambda x: x.replace('||', ''))


#%%
metrics_dict = {'levenshtein':levenshtein_loss, 'jaccard':jaccard_similarity, 'bleu':bleu_score, 'rouge':rouge_score}

idx = 10
row = df.iloc[idx]
# real = row['gt']
# pred = row['pred']
real = row['opes_gt']
pred = row['pred']

print('real-real:\n', f'{real}\n{real}')
for key, metric in metrics_dict.items():
    print(key, metric(real, real))

print('real-pred:\n', f'{real}\n{pred}')
for key, metric in metrics_dict.items():
    print(key, metric(real, pred))

similarity_reactants, similarity_products, overall_similarity = equation_similarity(real, pred, whole_equation=True, split='==')
print(f"(average) Reactants Similarity: {similarity_reactants:.2f}, Products Similarity: {similarity_products:.2f}, Overall Similarity: {overall_similarity:.2f}")


#%%
import random
import string

def add_noise(eq, noise_level=1):
    """
    Gradually adds noise to the string by applying random character substitution, insertion, deletion, or swapping.

    Args:
        eq (str): The original string.
        noise_level (int): Number of modifications to apply.

    Returns:
        str: The modified string with added noise.
    """
    noisy_eq = list(eq)
    
    for _ in range(noise_level):
        # Randomly choose a noise operation
        operation = random.choice(['substitute', 'insert', 'delete', 'swap'])

        if operation == 'substitute':
            # Random character substitution
            idx = random.randint(0, len(noisy_eq) - 1)
            noisy_eq[idx] = random.choice(string.ascii_letters + string.digits + ' ')
        
        elif operation == 'insert':
            # Random character insertion
            idx = random.randint(0, len(noisy_eq))
            noisy_eq.insert(idx, random.choice(string.ascii_letters + string.digits + ' '))
        
        elif operation == 'delete' and len(noisy_eq) > 0:
            # Random character deletion
            idx = random.randint(0, len(noisy_eq) - 1)
            noisy_eq.pop(idx)
        
        elif operation == 'swap' and len(noisy_eq) > 1:
            # Random character swap
            idx1 = random.randint(0, len(noisy_eq) - 2)
            noisy_eq[idx1], noisy_eq[idx1 + 1] = noisy_eq[idx1 + 1], noisy_eq[idx1]
    
    return ''.join(noisy_eq)

#%%
eq = real
print("Original:", eq)

# Apply gradual noise
metrics_dict = {'jaccard':jaccard_similarity, 'bleu':bleu_score}
# for key, metric in metrics_dict.items():
m_list = []
range_idx = range(1, 100)
for i in range_idx:  # Increase noise level gradually
    eq_ = add_noise(eq, noise_level=i)
    if not '==' in eq_ :
        continue
    print(f"With noise level {i}: {eq_}")
    # m = metric(real, eq_)
    similarity_reactants, similarity_products, overall_similarity = equation_similarity(real, eq_, whole_equation=True, split='==')
    m = overall_similarity
    m_list.append(m)
# plt.plot(range_idx, m_list, label='tanimoto')
plt.plot(m_list, label='tanimoto')
plt.legend()
    
            



#%%