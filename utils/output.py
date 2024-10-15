good_colors = {'orange': "#e69f00", 'sky': "#56b4e9", 'green': "#009e73",
               'yellow': "#f0e473", 'blue': "#0072b2", 'red': "#d55e00", 'pink': "#cc79a7"
}

adjust_l2r, adjust_r2l, adjust_t2c = len(r" $\rightarrow$ ") - len(r"->"), len(r" $\leftarrow$ ") - len(r"<-"), len(r"$||$") - len(r"||")
# adjust_len = {'LHS2RHS':adjust_l2r, 'RHS2LHS':adjust_r2l, 'TGT2CEQ':adjust_t2c, 'LHSOPE2RHS':adjust_l2r, 'RHSOPE2LHS':adjust_r2l, 'TGT2OPE2CEQ':adjust_t2c}
adjust_len = {'LHS2RHS': -1, 'RHS2LHS': -1, 'TGT2CEQ': 0, 'LHSOPE2RHS': -1, 'RHSOPE2LHS': -1, 'TGTOPE2CEQ': 0}
adjust_space = {'LHS2RHS': '', 'RHS2LHS': '', 'TGT2CEQ': ' ', 'LHSOPE2RHS': '', 'RHSOPE2LHS': '', 'TGTOPE2CEQ': ' '}

def generate_latex_table(filename, task, pred_color, data):
    replace_dict = {
        '->': r'$\rightarrow$', '<-': r'$\leftarrow$', '=>': r'$\Rightarrow$', '<=': r'$\Leftarrow$', 
        '<->': r'$\leftrightarrow$', '<=>': r'$\Leftrightarrow$', '==>': r'$\Longrightarrow$', 
        '<==': r'$\Longleftarrow$', '==': r'$\equiv$', '->>': r'$\longrightarrow$', '<<-': r'$\longleftarrow$', 
        '>>': r'$\gg$', '<<': r'$\ll$', '>>>': r'$\ggg$', '<<<': r'$\lll$', '<': r'$<$', '>': r'$>$', 
        '<=': r'$\leq$', '>=': r'$\geq$', '!=': r'$\neq$', '==': r'$\equiv$', '*': r'$*$', '||': r'$||$', 
        '&&': r'$\&$', '&': r'$\&$', 'δ': r'$\delta$', 'Δ': r'$\Delta$', 'α': r'$\alpha$', 'β': r'$\beta$', 
        'γ': r'$\gamma$', 'θ': r'$\theta$', 'ѳ': r'$\theta$', 'ω': r'$\omega$', "х": "x", "、": ","
    }

    table_caption = f"Selected examples of {task.upper()}"
    table_label = f"tab_s_{task.lower()}"

    with open(filename, 'w') as f:
        # Write LaTeX table header
        f.write(r"\begin{longtable}{c|c|p{10cm}|c|c}" + '\n')
        f.write(rf"\caption{{{table_caption}}}\label{{{table_label}}}\\" + '\n')
        f.write(r"No. &  Model & Text & GTS & JS \\" + '\n')
        f.write(r"\Xhline{5\arrayrulewidth}" + '\n')
        f.write(r"\endfirsthead" + '\n')
        f.write(r"\multicolumn{4}{c}{{\tablename\ \thetable{} -- continued from previous page}} \\" + '\n')
        f.write(r"No. &  Model & Text & GTS & JS \\" + '\n')
        f.write(r"\hline" + '\n')
        f.write(r"\endhead" + '\n')
        f.write(r"\hline" + '\n')
        f.write(r"\multicolumn{4}{r}{{Continued on next page}} \\" + '\n')
        f.write(r"\endfoot" + '\n')
        f.write(r"\hline" + '\n')
        f.write(r"\endlastfoot" + '\n')

        # Write table rows
        for i, row in data.iterrows():
            label = row['label']
            tan = round(row['gts.1'], 3)
            jac = round(row['js.1'], 3)
            tan0 = round(row['gts.0'], 3)
            jac0 = round(row['js.0'], 3)

            t_text = str(row['t_text']).strip()
            p_text = str(row['p_text.1']).strip()
            p_text0 = str(row['p_text.0']).strip()
            t_eq = str(row['t_eq']).strip()
            p_eq = str(row['p_eq.1']).strip()
            p_eq0 = str(row['p_eq.0']).strip()

            # Replace symbols using replace_dict
            l_adjust = 0
            for k, v in replace_dict.items():
                l_adjust += label.count(k) * (len(v) - len(k))
                t_text = t_text.replace(k, v)
                p_text = p_text.replace(k, v)
                p_text0 = p_text0.replace(k, v)
                t_eq = t_eq.replace(k, v)
                p_eq = p_eq.replace(k, v)
                p_eq0 = p_eq0.replace(k, v)

            len_label = len(label) + l_adjust + adjust_len[task.upper()]

            # Apply color to predictions if specified
            if pred_color is not None:
                p_text = f"{p_text[:len_label]} \\textcolor{{{pred_color}}}{{{p_text[len_label:].strip()}}}"
                p_text0 = f"{p_text0[:len_label]}{adjust_space[task.upper()]} \\textcolor{{{pred_color}}}{{{p_text0[len_label:].strip()}}}"

            # Construct table lines
            t_line = f"{i+1} & Ground truth & {t_text} & {1.0} & {1.0} \\\\"
            p_line = f"{i+1} & w/ fine-tuning & {p_text} & {tan} & {jac} \\\\"
            p_line0 = f"{i+1} & w/o fine-tuning & {p_text0} & {tan0} & {jac0} \\\\"

            # Escape '%' in text to prevent LaTeX errors
            for line in [t_line, p_line, p_line0]:
                line = line.replace('%', r"\%")

            # Write to file
            f.write(t_line + '\n')
            f.write(r"\hline" + '\n')
            f.write(p_line + '\n')
            f.write(r"\hline" + '\n')
            f.write(p_line0 + '\n')
            f.write(r"\Xhline{5\arrayrulewidth}" + '\n')

        # End the longtable environment
        f.write(r"\end{longtable}" + '\n')
    
    

