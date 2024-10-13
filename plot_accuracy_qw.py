#%%
"""
https://www.notion.so/240915-Preparing-main-figures-and-tables-10339215a12980aaa08bc92038ae737a?pvs=4#10339215a12980a19cc3f826a4af1a8c
"""
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

model_name = 'TGT2CEQ'
tag = 'qw'
fsize = 16
line_width = 3
msize = 30
ylim = [-0.02, 1.02]
color1, color2 = good_colors['blue'], good_colors['pink']

df_file = f'{model_name.lower()}_dgpt2_v1.2.1_test_1948_{tag}_df.csv'
df = pd.read_csv(join(save_dir, df_file))
# tan_mean, jac_mean, tan_mean0, jac_mean0 = df['acc'].mean(), df['jac'].mean(), df['acc.0'].mean(), df['jac.0'].mean()
tan, jac, tan0, jac0 = df['acc'], df['jac'], df['acc.0'], df['jac.0']
qw = df['qw']
tan_mean, jac_mean, tan_mean0, jac_mean0, qw_mean = tan.mean(), jac.mean(), tan0.mean(), jac0.mean(), qw.mean()
acc_mean_dict = {'tan': [tan_mean, tan_mean0], 'jac': [jac_mean, jac_mean0]}
acc_dict = {'tan': [tan, tan0], 'jac': [jac, jac0]}
metric_name_dict = {'tan': 'GTS', 'jac': 'JS'}
print(f'{model_name} tan mean: {tan_mean:.3f}, jac mean: {jac_mean:.3f}')
print(f'{model_name} tan0 mean: {tan_mean0:.3f}, jac0 mean: {jac_mean0:.3f}')

# read the csv file. givent the column names ('element', 'acuracy'), the csv file is read and stored in a pandas dataframe 
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
for i, (ax, eval_) in enumerate(zip(axs, ['tan', 'jac'])):
    metric_name = metric_name_dict[eval_]
    # csv_file1 = f'{csv_file1_header}_{eval_}.csv'
    # csv_file2 = f'{csv_file2_header}_{eval_}.csv'
    # df1 = pd.read_csv(join(save_dir, csv_file1), names=['element', 'accuracy'])
    # df2 = pd.read_csv(join(save_dir, csv_file2), names=['element', 'accuracy'])
    for j, c in enumerate([color1, color2]):
        # get the atomic number of the element by getting the index of the element in chemical_symbols list
        # df_['atomic_number'] = df_['element'].apply(lambda x: chemical_symbols.index(x)+1)
        # ax.scatter(df_['atomic_number'], df_['accuracy'], color=c, s=msize, label=f'{model_name} {eval} {j+1}')
        # scatter plot of quantum weight (qw) against the accuracy
        acc = acc_dict[eval_][j]
        ax.scatter(qw, acc, color=c, s=msize, label=f'{model_name} {eval_} {j+1}')
        # draw horizontal line at y = 0.5
        ax.axhline(acc_mean_dict[eval_][j], color=c, linestyle='-', linewidth=line_width, alpha=0.4)
        ax.set_ylim(ylim)
        ax.set_xlabel('$K_{xx}$', fontsize=fsize)
        ax.set_ylabel(f'Accuracy ({eval_})', fontsize=fsize)
        # set the xticks size 
        ax.tick_params(axis='x', labelsize=fsize)
        # set the yticks size
        ax.tick_params(axis='y', labelsize=fsize)
        ax.set_title(f'{model_name} ({metric_name})', fontsize=fsize)
    fig.suptitle(model_name, fontsize=fsize)
# fig.savefig(join(fig_dir, f'{model_name}_tan_jac.pdf'))
# %%
