#%%
import re
import numpy as np
from scipy.optimize import linear_sum_assignment


# def parse_formula(formula):
def element_vector(formula):
    # Adjusted regex pattern to match element symbols followed by optional integers or decimal numbers
    elements = re.findall('([A-Z][a-z]*)(\d*\.?\d+)?', formula)
    return {element: float(count) if count else 1 for element, count in elements}

# def element_vector(formula):
#     """
#     Convert a chemical formula into a vector (dictionary) of element counts.
#     """
#     element_counts = parse_formula(formula)  # Assuming parse_formula is defined as before
#     return element_counts

def tanimoto_similarity_elemental(formula1, formula2):
    """
    Calculate the Tanimoto similarity between two chemical compositions based on element counts.
    """
    # print('(formula1, formula2)', (formula1, formula2))
    vec1 = element_vector(formula1)
    vec2 = element_vector(formula2)
    # print('(vec1, vec2) ', (vec1, vec2))
    
    common_elements = set(vec1.keys()) & set(vec2.keys())
    if not common_elements:
        return 0  # No similarity if no elements in common
    
    # Calculate the dot product
    dot_product = sum(vec1[elem] * vec2[elem] for elem in common_elements)
    
    # Calculate the sum of squares
    sum_squares1 = sum(count ** 2 for count in vec1.values())
    sum_squares2 = sum(count ** 2 for count in vec2.values())
    
    # Calculate Tanimoto similarity
    # print('sum_squares1, sum_squares2, dot_product', sum_squares1, sum_squares2, dot_product)
    similarity = dot_product / (sum_squares1 + sum_squares2 - dot_product)
    
    return similarity

def split_half(equation):   #!TODO: check thhe code
    # Calculate the midpoint of the string
    midpoint = len(equation) // 2
    # Find the nearest space to the midpoint
    # Search for the nearest space character to the left of the midpoint
    left_index = equation.rfind(' ', 0, midpoint)   #!
    # Search for the nearest space character to the right of the midpoint
    right_index = equation.find(' ', midpoint)  #!
    # Determine which space is closer to the midpoint
    if midpoint - left_index < right_index - midpoint:
        split_index = left_index
    else:
        split_index = right_index
    # Split the equation into two parts
    part1 = equation[:split_index].strip()
    part2 = equation[split_index:].strip()
    return part1, part2

def split_equation(equation, split):
    if split in equation:
        reactants_part, products_part = equation.split(split, 1)
        if len(reactants_part)*5 < len(products_part):
            reactants_part, products_part = split_half(equation)
    else:
        reactants_part, products_part = split_half(equation)
    reactants = [reactant.strip() for reactant in reactants_part.split("+")]
    products = [product.strip() for product in products_part.split("+")]
    return reactants, products

def compare_formula_lists(formula_list1, formula_list2):
    # Compute similarity matrix
    similarity_matrix = [
        [tanimoto_similarity_elemental(formula1, formula2) for formula2 in formula_list2] 
        for formula1 in formula_list1
    ]
    
    # Hungarian algorithm to find the best pairing
    row_ind, col_ind = linear_sum_assignment(cost_matrix=-1 * np.array(similarity_matrix))
    
    # Calculate overall similarity
    similarity = sum(similarity_matrix[row][col] for row, col in zip(row_ind, col_ind)) / len(formula_list1)#max(len(formula_list1), len(formula_list2))
    
    return similarity

def equation_similarity_partial(equation1, equation2, whole_equation=True, split='->'):
    if whole_equation:
        reactants1, products1 = split_equation(equation1, split)
        reactants2, products2 = split_equation(equation2, split)
     
        similarity_reactants = compare_formula_lists(reactants1, reactants2)
        similarity_products = compare_formula_lists(products1, products2)
        
        overall_similarity = (similarity_reactants + similarity_products) / 2
    else:
        formula_list1 = equation1.split("+")
        formula_list2 = equation2.split("+")

        # print('formula_list1, formula_list2: ', formula_list1, formula_list2)
        overall_similarity = compare_formula_lists(formula_list1, formula_list2)
        similarity_reactants = similarity_products = overall_similarity  # In this case, they are the same
    
    return similarity_reactants, similarity_products, overall_similarity
        
def equation_similarity(equation1, equation2, whole_equation=True, split='->'):
    # Calculate similarity in both directions
    sim_r1, sim_p1, sim1 = equation_similarity_partial(equation1, equation2, whole_equation, split)
    sim_r2, sim_p2, sim2 = equation_similarity_partial(equation2, equation1, whole_equation, split)
    sim_r, sim_p, sim = (sim_r1+sim_r2)/2, (sim_p1+sim_p2)/2, (sim1+sim2)/2
    return sim_r, sim_p, sim


def jaccard_similarity(str1, str2):
    set1 = set(str1.split())
    set2 = set(str2.split())
    return len(set1.intersection(set2)) / len(set1.union(set2))

def jaccard_similarity_wo_sym(str1, str2, arrow='->'):
    set1 = set(str1.split())
    set2 = set(str2.split())
    ####
    # remove '+' from the set
    set1.discard('+')
    set1.discard(arrow)
    set2.discard('+')
    set2.discard(arrow)
    ####
    return len(set1.intersection(set2)) / len(set1.union(set2))


# def find_atomic_species(formula):   #TODO: could we combine this with the element_vector function?. Also check if ''fformula' is correct. 
#     # Regular expression pattern for element symbols: one uppercase letter followed by an optional lowercase letter
#     pattern = r'[A-Z][a-z]?'
    
#     # Find all occurrences of the pattern in the formula string
#     elements = re.findall(pattern, formula)
    
#     # Remove duplicates by converting the list to a set, then convert back to a list if needed
#     unique_elements = list(set(elements))
#     # print('unieuq_elements: ', unique_elements)
#     chem_list = []
#     for el in unique_elements:
#         if el in chemical_symbols:
#             chem_list.append(el)
#         else: 
#             if el[0] in chemical_symbols:
#                 chem_list.append(el[0])
 
#     return list(set(chem_list))

# def save_dict_as_csv(dictionary, filename):
#     with open(filename, 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         # Write the header row, if needed
#         # writer.writerow(['Element', 'Value'])
#         for key, value in dictionary.items():
#             writer.writerow([key, value])
