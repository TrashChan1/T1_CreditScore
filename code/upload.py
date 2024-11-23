import os
from pathlib import Path
import pandas as pd
import logging

# Load data from a folder clled data within my project file
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

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(file_name: str = "credit_score_data.csv", folder_name: str = "data"):
    """
    Load data from the specified csv file within the project directory.
    
    Parameters:
        file_name (str): The name of the file to load
        folder_name (str): The folder where the data is stored
    
    Returns:
        pd.DataFrame or None: The loaded DataFrame if successful, None if an error occurs
    """
    try:
        current_dir = Path.cwd()
        data_folder = current_dir.parent / folder_name
        file_path = data_folder / file_name
        
        if not file_path.exists():
            logger.error(f"Error: File not found at '{file_path}'")
            return None
            
        df = pd.read_csv(file_path)
        
        if df.empty:
            logger.error("Error: The file is empty")
            return None
            
        logger.info(f"File loaded: {file_path}")
        return df
        
    except pd.errors.EmptyDataError:
        logger.error("Error: The file is empty")
    except pd.errors.ParserError:
        logger.error("Error: The file is corrupted or has an invalid format")
    except Exception as e:
        logger.error(f"Error: Unable to load the file - {str(e)}")
    
    return None

def handle_load_data():
    """
    Handle the load data menu option with retry capabilities.
    
    Returns:
        pd.DataFrame or None: The loaded DataFrame if successful, None if user chooses to return to menu
    """
    while True:
        print("\nAttempting to load data file...")
        
        df = load_data()
        
        if df is not None:
            print(f"Dataset loaded: {len(df):,} rows, {len(df.columns)} columns")
            return df
        
        print("\nWhat would you like to do?")
        print("(1) Try loading the same file again")
        print("(2) Try loading a different file")
        print("(3) Return to main menu")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            continue
        elif choice == '2':
            new_file = input("Enter the name of the file to load: ")
            df = load_data(file_name=new_file)
            if df is not None:
                print(f"Dataset loaded: {len(df):,} rows, {len(df.columns)} columns")
                return df
        elif choice == '3':
            return None
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    print("Testing upload module...")
    result = handle_load_data()
    if result is not None:
        print("Test complete")
    else:
        print("Returned to main menu")
