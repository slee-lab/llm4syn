#%%
import os 
from os.path import join
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
plt.rcParams['font.family'] = ['Arial', 'sans-serif']


from utils.metrics import *
from utils.data import *
from utils.plot_data import *


#%%

save_dir = './save'
fig_dir = './figures'

model_name1 = 'TGT2CEQ'
tag = 'v3.1'
model_name2 = model_name1[:3] + 'OPE' + model_name1[3:]
csv_file1_header = f'{model_name1.lower()}_dgpt2_v1.2.1_test_1948_{tag}' 
csv_file2_header = f'{model_name2.lower()}_dgpt2_v1.2.1_test_1948_{tag}'    # 'lhsope2rhs_dgpt2_v1.2.1_test_1948_v2.1_tan.csv'
fsize = 16
line_width = 3
msize = 30
ylim = [-0.02, 1.02]
color1, color2 = good_colors['blue'], good_colors['orange']

df_file1 = f'{model_name1.lower()}_dgpt2_v1.2.1_test_1948_{tag}_df.csv'
df_file2 = f'{model_name2.lower()}_dgpt2_v1.2.1_test_1948_{tag}_df.csv'
df1 = pd.read_csv(join(save_dir, df_file1))
df2 = pd.read_csv(join(save_dir, df_file2))
tan_mean1, jac_mean1, tan_mean2, jac_mean2 = df1['acc'].mean(), df1['jac'].mean(), df2['acc'].mean(), df2['jac'].mean()
acc_dict = {'tan': [tan_mean1, tan_mean2], 'jac': [jac_mean1, jac_mean2]}
metriic_name_dict = {'tan': 'Tanimoto', 'jac': 'Jaccard'}
print(f'{model_name1} tan mean: {tan_mean1:.3f}, jac mean: {jac_mean1:.3f}')
print(f'{model_name2} tan0 mean: {tan_mean2:.3f}, jac0 mean: {jac_mean2:.3f}')

# read the csv file. givent the column names ('element', 'acuracy'), the csv file is read and stored in a pandas dataframe 
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
for i, (ax, eval_) in enumerate(zip(axs, ['tan', 'jac'])):
    metric_name = metriic_name_dict[eval_]
    csv_file1 = f'{csv_file1_header}_{eval_}.csv'
    csv_file2 = f'{csv_file2_header}_{eval_}.csv'
    df1 = pd.read_csv(join(save_dir, csv_file1), names=['element', 'accuracy'])
    df2 = pd.read_csv(join(save_dir, csv_file2), names=['element', 'accuracy'])
    for j, (df_, c) in enumerate(zip([df1, df2], [color1, color2])):
        # get the atomic number of the element by getting the index of the element in chemical_symbols list
        df_['atomic_number'] = df_['element'].apply(lambda x: chemical_symbols.index(x)+1)
        ax.scatter(df_['atomic_number'], df_['accuracy'], color=c, s=msize)#, label=f'{model_name} {eval} {j+1}')
        # draw horizontal line at y = 0.5
        ax.axhline(acc_dict[eval_][j], color=c, linestyle='-', linewidth=line_width, alpha=0.5)
        ax.set_ylim(ylim)
        ax.set_xlabel('Atomic number', fontsize=fsize)
        ax.set_ylabel('Accuracy', fontsize=fsize)
        # set the xticks size 
        ax.tick_params(axis='x', labelsize=fsize)
        # set the yticks size
        ax.tick_params(axis='y', labelsize=fsize)
        ax.set_title(f'{model_name1} vs {model_name2} ({metric_name})', fontsize=fsize)
    fig.suptitle(f'{model_name1} vs {model_name2}', fontsize=fsize)
fig.savefig(join(fig_dir, f'{model_name1}_{model_name2}_tan_jac.pdf'))