import sys
import utils
import os
import json

# Set the root directory
current_file_path = os.path.abspath(__file__)

# Get the root directory
root_dir = os.path.dirname(os.path.dirname(current_file_path))

# Add the directory containing 'modeling.py' to the Python path
modeling_dir = os.path.join(root_dir, 'Dataset Creation and Feature Extraction')
sys.path.append(modeling_dir)

# Now you can import 'modeling'
import modeling

# script.py
if len(sys.argv) < 1:
    print("Not enough arguments")

data = modeling.predict(sys.argv[1])

converted_data = utils.convert_ndarray_to_list(data)

# Save the data to a JSON file
with open("data/input.json", "w") as json_file:
    json.dump(converted_data, json_file, indent=4)

bool_matrix = utils.parse_json_file()

utils.save_boolean_matrix(bool_matrix)

utils.save_boolean_matrix_interactive(bool_matrix)

utils.generate_waveform()