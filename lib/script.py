import sys
import utils
import os

# Set the root directory
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Construct the absolute path to the directory containing modeling.py
module_dir = os.path.join(root_dir, 'Dataset Creation and Feature Extraction')

# Add the module directory to sys.path
sys.path.insert(module_dir)

# Import the module
import modeling

# script.py
if len(sys.argv) < 1:
    print("Not enough arguments")

data = modeling.predict(sys.argv[1])

bool_matrix = utils.parse_json_file(data)

utils.save_boolean_matrix(bool_matrix)

utils.save_boolean_matrix_interactive(bool_matrix)

utils.generate_waveform(data, bool_matrix)