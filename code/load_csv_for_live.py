import os
from pathlib import Path
import pandas as pd


# Load data from a folder called data within my project file
#  .. my_project
#     |
#     |___code
#     |   |
#     |   |__ CS3500_Starter_Notebook.ipynb
#     |
#     |___data
#         |
#         |__ credit_score.csv
#
#---------------------------------------------------------------

# Get the current working directory
current_dir = os.getcwd() 

# Construct a path to the parent directory
parent_dir = os.path.dirname(current_dir)

# Access a file in the parent directory
file_path = os.path.join(parent_dir, "data")
file_path = os.path.join(file_path, "credit_score_data.csv")

# Load Credit Score data
df = pd.read_csv(file_path) 


