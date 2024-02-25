from collections import defaultdict
import re

def parse_formula_to_vector(formula):
    """Parse a chemical formula into a vector (dict) of element counts."""
    elements = re.findall('([A-Z][a-z]*)(\d*)', formula)
    vector = defaultdict(int)
    for element, count in elements:
        vector[element] += int(count) if count else 1
    return vector

def tanimoto_similarity(formula1, formula2):
    """Calculate the Tanimoto similarity between two chemical formulas."""
    vec1 = parse_formula_to_vector(formula1)
    vec2 = parse_formula_to_vector(formula2)
    
    # Calculate the dot product
    dot_product = sum(vec1[elem] * vec2[elem] for elem in vec1 if elem in vec2)
    
    # Calculate the sum of squares
    sum_squares1 = sum(count ** 2 for count in vec1.values())
    sum_squares2 = sum(count ** 2 for count in vec2.values())
    
    # Calculate Tanimoto similarity
    similarity = dot_product / (sum_squares1 + sum_squares2 - dot_product)
    
    return similarity

# Example usage
formula1 = 'Na2CO3'
formula2 = 'K2CO3'

similarity = tanimoto_similarity(formula1, formula2)
print(f"Tanimoto similarity between {formula1} and {formula2}: {similarity:.3f}")


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


