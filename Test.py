from sympy.logic.boolalg import truth_table, Or, And, Not
from sympy import symbols
from sympy.logic.boolalg import to_dnf
from itertools import product

from itertools import product
import random

# Define symbolic variables x_0 to x_7
x = symbols('x:16')

def simplify_boolean_function(truth_table):
    num_inputs = len(truth_table[0]) - 1

    # Use the first num_inputs variables from x as input variables
    input_variables = x[:num_inputs]

    boolean_function = Or(*[And(*[var if bit else Not(var) for var, bit in zip(input_variables, row[:-1])]) for row in truth_table if row[-1]])
    # Get the truth table of the boolean function
    

    simplified_boolean_function = to_dnf(boolean_function, simplify=True, force= True)

    return simplified_boolean_function

# Example usage:
#truth_table = [(0, 0,0,0, 1,0, 0,0,0, 1), (0, 0,0,1, 1,0, 0,0,0, 1), (1, 0,0,0, 0,0, 0,0,0, 1), (1, 0,0,1, 1,0, 0,0,0, 1)]
# Number of bits
num_bits = 3

# Generate all possible 16-bit binary sequences
binary_sequences = list(product([0, 1], repeat=num_bits))
#print(binary_sequences)
# Assign random labels (0 or 1) to each sequence
#truth_table = [(seq + (random.choice([0, 1]),)) for seq in binary_sequences]
truth_table = [(seq + (seq[0]*seq[1],)) for seq in binary_sequences]
print("HIII")
print(truth_table)

simplified_function = simplify_boolean_function(truth_table)
print(f"Simplified boolean function: {simplified_function}")
