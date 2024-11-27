from pathlib import Path
import pandas as pd
from datetime import datetime
import time
import os
import chardet

def get_current_time():
    """Return current time in the specified format"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def load_data(file_name: str = "credit_score_data.csv", folder_name: str = "data"):
    """
    Load data from the specified CSV file within the project directory.
    
    Parameters:
        file_name (str): The name of the file to load
        folder_name (str): The folder where the data is stored
    
    Returns:
        pd.DataFrame or None: The loaded DataFrame if successful, None if an error occurs
        float: Time taken to load the data
    """
    try:
        print("\nLoading and cleaning input data set:")
        print("************************************")
        print(f"[{get_current_time()}] Starting Script")
        print(f"[{get_current_time()}] Loading training data set")
        
        # Start timing
        start_time = time.time()
        
        # Load the data
        current_dir = Path.cwd()
        data_folder = current_dir.parent / folder_name
        file_path = data_folder / file_name
        
        # Check if file exists
        if not file_path.exists():
            print(f"[{get_current_time()}] Error: File not found at '{file_path}'")
            return None, 0
            
        # Check file size
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
        if file_size > 1000:  # Example: 1GB limit
            print(f"[{get_current_time()}] Error: File size ({file_size:.2f}MB) exceeds maximum allowed size")
            return None, 0
            
        # Check permissions
        if not os.access(file_path, os.R_OK):
            print(f"[{get_current_time()}] Error: Permission denied - Cannot read file")
            return None, 0
            
        # Check if file is empty
        if file_size == 0:
            print(f"[{get_current_time()}] Error: File is empty")
            return None, 0
            
        # Detect file encoding
        with open(file_path, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']
        
        # Try to load the CSV file
        df = pd.read_csv(file_path, encoding=encoding)
        
        if df.empty:
            print(f"[{get_current_time()}] Error: No data found in file")
            return None, 0
            
        # Basic format validation
        if len(df.columns) < 2:  # Assuming we expect at least 2 columns
            print(f"[{get_current_time()}] Error: Invalid file format - Incorrect number of columns")
            return None, 0
        
        # Calculate loading time
        load_time = time.time() - start_time
        
        # Print the required information
        print(f"[{get_current_time()}] Total Columns Read: {len(df.columns)}")
        print(f"[{get_current_time()}] Total Rows Read: {len(df)}")
        print(f"\nTime to load is: {load_time:.2f} seconds")
        
        return df, load_time
        
    except PermissionError:
        print(f"[{get_current_time()}] Error: Permission denied - Cannot access file")
    except pd.errors.EmptyDataError:
        print(f"[{get_current_time()}] Error: The file is empty")
    except pd.errors.ParserError:
        print(f"[{get_current_time()}] Error: The file is corrupted or has an invalid format")
    except UnicodeDecodeError:
        print(f"[{get_current_time()}] Error: File encoding not supported")
    except MemoryError:
        print(f"[{get_current_time()}] Error: File too large to process")
    except Exception as e:
        print(f"[{get_current_time()}] Error: Unable to load the file - {str(e)}")
    
    return None, 0

def handle_load_data():
    """
    Handle the load data menu option with retry capabilities.
    
    Returns:
        pd.DataFrame or None: The loaded DataFrame if successful, None if user chooses to return to menu
    """
    df, load_time = load_data()
    
    while True:
        if df is not None:
            return df
            
        print("\nWhat would you like to do?")
        print("(1) Try loading the same file again")
        print("(2) Try loading a different file")
        print("(3) Return to main menu")
        
        while True:
            choice = input("Enter your choice (1-3): ")
            
            if choice == '1':
                df, load_time = load_data()
                break
            elif choice == '2':
                new_file = input("Enter the name of the file to load: ")
                df, load_time = load_data(file_name=new_file)
                break
            elif choice == '3':
                return None
            else:
                print("Invalid choice. Please try again.")

if __name__ == "__main__":
    result = handle_load_data()
