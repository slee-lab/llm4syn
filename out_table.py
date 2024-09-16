#%%
import os 
from os.path import join
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt

from utils.metrics import *
from utils.data import *


#%%

save_dir = './save'
model_name = 'TGTOPE2CEQ'
df_file = f'{model_name.lower()}_dgpt2_v1.2.1_test_1948_v3.1_df.csv'
len_out = 1900

adjust_l2r, adjust_r2l, adjust_t2c = len(r" $\rightarrow$ ") - len(r"->"), len(r" $\leftarrow$ ") - len(r"<-"), len(r"$||$") - len(r"||")
# adjust_len = {'LHS2RHS':adjust_l2r, 'RHS2LHS':adjust_r2l, 'TGT2CEQ':adjust_t2c, 'LHSOPE2RHS':adjust_l2r, 'RHSOPE2LHS':adjust_r2l, 'TGT2OPE2CEQ':adjust_t2c}
adjust_len = {'LHS2RHS': -1, 'RHS2LHS': -1, 'TGT2CEQ': 0, 'LHSOPE2RHS': -1, 'RHSOPE2LHS': -1, 'TGTOPE2CEQ': 0}
adjust_space = {'LHS2RHS': '', 'RHS2LHS': '', 'TGT2CEQ': ' ', 'LHSOPE2RHS': '', 'RHSOPE2LHS': '', 'TGTOPE2CEQ': ' '}

# load the dataframe
df = pd.read_csv(join(save_dir, df_file))
# sort the dataframe by accuracy 'ja' and the 'acc' 
df = df.sort_values(by=['acc'], ascending=False)
# df = df.sort_values(by=['jac', 'acc'], ascending=False)


pred_color = 'blue'

def generate_latex_table(filename, table_caption, table_label, data):
    # need_doll = ['->', '<-', '=>', '<=', '<->', '<=>', '==>', '<==', '==', '->>', '<<-', '>>', '<<', '>>>', '<<<', '>>>',]
    replace_dict = {'->': r'$\rightarrow$', '<-': r'$\leftarrow$', '=>': r'$\Rightarrow$', '<=': r'$\Leftarrow$', 
                    '<->': r'$\leftrightarrow$', '<=>': r'$\Leftrightarrow$', '==>': r'$\Longrightarrow$', 
                    '<==': r'$\Longleftarrow$', '==': r'$\equiv$', '->>': r'$\longrightarrow$', '<<-': r'$\longleftarrow$', 
                    '>>': r'$\gg$', '<<': r'$\ll$', '>>>': r'$\ggg$', '<<<': r'$\lll$', '>>>': r'$\ggg$',
                    '<': r'$<$', '>': r'$>$', '<=': r'$\leq$', '>=': r'$\geq$', '!=': r'$\neq$', '==': r'$\equiv$', 
                    '*': r'$*$', '||': r'$||$', '&&': r'$\&$', '&': r'$\&$', '==': r'$\equiv$',
                    'δ': r'$\delta$', 'Δ': r'$\Delta$', 'α': r'$\alpha$', 'β': r'$\beta$', 'γ': r'$\gamma$', 'θ': r'$\theta$', 'ѳ': r'$\theta$',
                    "х": "x", "、":","}
    with open(filename, 'w') as f:
        # Write the LaTeX table header using longtable
        f.write(r"\begin{longtable}{c|c|p{10cm}|c|c}" + '\n')
        f.write(r"\caption{" + table_caption + r"}\label{" + table_label + r"}\\" + '\n')
        f.write(r"No. &  Model & Text & TANI & JAC \\" + '\n')
        f.write(r"\Xhline{5\arrayrulewidth}" + '\n')
        # f.write(r"\hline" + '\n')
        f.write(r"\endfirsthead" + '\n')
        f.write(r"\multicolumn{4}{c}%" + '\n')
        f.write(r"{{\tablename\ \thetable{} -- continued from previous page}} \\" + '\n')
        f.write(r"No. &  Model & Text & TANI & JAC \\" + '\n')
        # f.write(r"\Xhline{5\arrayrulewidth}" + '\n')
        f.write(r"\hline" + '\n')
        f.write(r"\endhead" + '\n')
        # f.write(r"\Xhline{5\arrayrulewidth}" + '\n')
        f.write(r"\hline" + '\n')
        f.write(r"\multicolumn{4}{r}{{Continued on next page}} \\" + '\n')
        f.write(r"\endfoot" + '\n')
        # f.write(r"\Xhline{5\arrayrulewidth}" + '\n')
        f.write(r"\hline" + '\n')
        f.write(r"\endlastfoot" + '\n')

        # Write the table rows
        for i in range(len(data)):
            print(i)
            row = data.iloc[i]
            # print(row)
            # print(row.keys())
            label = row['label']
            # len_label = len(label)  ##+ adjust_len[model_name]
            tan = round(row['acc'], 3)
            jac = round(row['jac'], 3)
            tan0 = round(row['acc.0'], 3)
            jac0 = round(row['jac.0'], 3)
            gt_text = row['gt_text']
            pr_text = row['pr_text']
            pr_text0 = row['pr_text.0']
            gt_eq = row['gt_eq']
            pr_eq = row['pr_eq']
            pr_eq0 = row['pr_eq.0']
            print('pr_text0:', pr_text0)
            print('pr_eq0:', pr_eq0)
        
            # print(pr_eq0)
            l_adjust = 0
            for k, v in replace_dict.items():
                # count how many times k appears in the text
                # print(k, v)
                l_adjust += label.count(k) * (len(v) - len(k))
                gt_text = gt_text.replace(k, v)
                pr_text = pr_text.replace(k, v)
                pr_text0 = pr_text0.replace(k, v)
                gt_eq = gt_eq.replace(k, v)
                pr_eq = pr_eq.replace(k, v)
                pr_eq0 = pr_eq0.replace(k, v)
            len_label = len(label) + l_adjust + adjust_len[model_name]
            if pred_color is not None:
                # gt_text = gt_text.replace(gt_eq, f"\\textcolor{{{pred_color}}}{{{gt_eq}}}")
                # pr_text = pr_text.replace(pr_eq, f"\\textcolor{{{pred_color}}}{{{pr_eq}}}")
                # pr_text0 = pr_text0.replace(pr_eq0, f"\\textcolor{{{pred_color}}}{{{pr_eq0}}}")
                pr_text = pr_text[:len_label] + ' ' + f"\\textcolor{{{pred_color}}}{{{pr_text[len_label:]}}}"
                pr_text0 = pr_text0[:len_label] + adjust_space[model_name] + f"\\textcolor{{{pred_color}}}{{{pr_text0[len_label:]}}}"
            gt_line = f"{i+1} & True & {gt_text} & {1.0} & {1.0} \\\\" + '\n'
            pr_line = f"{i+1} & {model_name} & {pr_text} & {tan} & {jac} \\\\" + '\n'
            pr_line0 = f"{i+1} & distilGPT2 & {pr_text0} & {tan0} & {jac0} \\\\" + '\n'
            if '%' in gt_line:
                gt_line = gt_line.replace('%', r"\%")
            if '%' in pr_line:
                pr_line = pr_line.replace('%', r"\%")
            if '%' in pr_line0:
                pr_line0 = pr_line0.replace('%', r"\%")
            f.write(gt_line)
            f.write(r"\hline" + '\n')
            f.write(pr_line)
            f.write(r"\hline" + '\n')
            f.write(pr_line0)
            f.write(r"\Xhline{5\arrayrulewidth}" + '\n')
            

        # End the longtable environment
        f.write(r"\end{longtable}" + '\n')

# Example usage
generate_latex_table(f"./tex/si_tab_{model_name.lower()}_{len_out}.tex", f"Selected examples of {model_name}", f"tab_s_{model_name.lower()}", df.iloc[:len_out])


