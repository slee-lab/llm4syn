
separator_dict = {'lhs2rhs': '->', 'rhs2lhs': '<-', 'tgt2ceq': '||', 'lhsope2rhs': '->', 'rhsope2lhs': '<-', 'tgtope2ceq': '||'}

out_conf_dict = {'lhs2rhs': {'type': 'mul', 'value': 2.1}, 
                 'rhs2lhs': {'type': 'mul', 'value': 2.1}, 
                 'tgt2ceq': {'type': 'mul', 'value': 4.0}, 
                 'lhsope2rhs': {'type': 'mul', 'value': 2.1}, 
                 'rhsope2lhs': {'type': 'mul', 'value': 2.1}, 
                 'tgtope2ceq': {'type': 'mul', 'value': 4.0}}

gpt_model_dict = {'dgpt2': "distilbert/distilgpt2" , 
                  'gpt2': "gpt2"}    

full_equation_dict = {'lhs2rhs': False, 'rhs2lhs': False, 'tgt2ceq': True, 'lhsope2rhs': False, 'rhsope2lhs': False, 'tgtope2ceq': True}