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

chemical_symbols = [

    # 1
    'H', 'He',
    # 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    # 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    # 7
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og']


def parse_formula(formula):
    # Adjusted regex pattern to match element symbols followed by optional integers or decimal numbers
    elements = re.findall('([A-Z][a-z]*)(\d*\.?\d+)?', formula)
    return {element: float(count) if count else 1 for element, count in elements}

def element_vector(formula):
    """
    Convert a chemical formula into a vector (dictionary) of element counts.
    """
    element_counts = parse_formula(formula)  # Assuming parse_formula is defined as before
    return element_counts

def tanimoto_similarity_elemental(comp1, comp2):
    """
    Calculate the Tanimoto similarity between two chemical compositions based on element counts.
    """
    # print('(comp1, comp2)', (comp1, comp2))
    vec1 = element_vector(comp1)
    vec2 = element_vector(comp2)
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

def split_half(equation):
    # Calculate the midpoint of the string
    midpoint = len(equation) // 2
    # Find the nearest space to the midpoint
    # Search for the nearest space character to the left of the midpoint
    left_index = equation.rfind(' ', 0, midpoint)
    # Search for the nearest space character to the right of the midpoint
    right_index = equation.find(' ', midpoint)
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

def compare_components(components1, components2):
    # Compute similarity matrix
    similarity_matrix = [
        [tanimoto_similarity_elemental(comp1, comp2) for comp2 in components2] 
        for comp1 in components1
    ]
    
    # Hungarian algorithm to find the best pairing
    row_ind, col_ind = linear_sum_assignment(cost_matrix=-1 * np.array(similarity_matrix))
    
    # Calculate overall similarity
    overall_similarity = sum(similarity_matrix[row][col] for row, col in zip(row_ind, col_ind)) / max(len(components1), len(components2))
    
    return overall_similarity

def equation_similarity_(equation1, equation2, whole_equation=True, split='->'):
    if whole_equation:
        # print('[0] equation1: ', equation1)
        # print('[0] equation2: ', equation2)
        # print('siplit: ', split)
        if split in equation1:
            reactants1, products1 = split_equation(equation1, split)
        else: 
            reactants1, products1 = split_half(equation1)
        if split in equation2:
            reactants2, products2 = split_equation(equation2, split)
        else: 
            reactants2, products2 = split_half(equation2)
        # print('[0] reactants1: ', reactants1)
        # print('[0] products1: ', products1)
        # print('[0] reactants2: ', reactants2)
        # print('[0] products2: ', products2) 
     
        similarity_reactants = compare_components(reactants1, reactants2)
        similarity_products = compare_components(products1, products2)
        
        overall_similarity = (similarity_reactants + similarity_products) / 2
    else:
        components1 = equation1.split("+")
        components2 = equation2.split("+")

        # print('components1, components2: ', components1, components2)
        overall_similarity = compare_components(components1, components2)
        similarity_reactants = similarity_products = overall_similarity  # In this case, they are the same
    
    return similarity_reactants, similarity_products, overall_similarity
        
def equation_similarity(equation1, equation2, whole_equation=True, split='->'):
    sim_r1, sim_p1, sim1 = equation_similarity_(equation1, equation2, whole_equation, split)
    sim_r2, sim_p2, sim2 = equation_similarity_(equation2, equation1, whole_equation, split)
    sim_r, sim_p, sim = (sim_r1+sim_r2)/2, (sim_p1+sim_p2)/2, (sim1+sim2)/2
    return sim_r, sim_p, sim

def find_atomic_species(formula):
    # Regular expression pattern for element symbols: one uppercase letter followed by an optional lowercase letter
    pattern = r'[A-Z][a-z]?'
    
    # Find all occurrences of the pattern in the formula string
    elements = re.findall(pattern, formula)
    
    # Remove duplicates by converting the list to a set, then convert back to a list if needed
    unique_elements = list(set(elements))
    # print('unieuq_elements: ', unique_elements)
    chem_list = []
    for el in unique_elements:
        if el in chemical_symbols:
            chem_list.append(el)
        else: 
            if el[0] in chemical_symbols:
                chem_list.append(el[0])
 
    return list(set(chem_list))

def save_dict_as_csv(dictionary, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write the header row, if needed
        # writer.writerow(['Element', 'Value'])
        for key, value in dictionary.items():
            writer.writerow([key, value])

def exact_match_accuracy(target, prediction):
    return target.strip() == prediction.strip()


def parse_equation(equation):
    reactants, products = equation.split("->")
    reactants_set = set(reactants.strip().split(" + "))
    products_set = set(products.strip().split(" + "))
    return reactants_set, products_set

def component_wise_accuracy(target, prediction):
    target_reactants, target_products = parse_equation(target)
    prediction_reactants, prediction_products = parse_equation(prediction)
    
    reactants_correct = target_reactants == prediction_reactants
    products_correct = target_products == prediction_products
    
    return reactants_correct and products_correct


