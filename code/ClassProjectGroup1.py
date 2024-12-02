
# General Packages
import math
import os
from pathlib import Path
from datetime import datetime
import time
import chardet


# data handling libraries
import pandas as pd
import numpy as np
from tabulate import tabulate

# visualization libraries
from matplotlib import pyplot as plt
import seaborn as sns

# extra libraries
import warnings
warnings.filterwarnings('ignore')

# Packages to support NN

# sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

#tensorflow
import tensorflow as tf
from tensorflow import keras

# Keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

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


def clean_data():

    df.isna().sum()


    # ### Dropping some columns

    # Dropping not related columns
    ##############################
    columns_to_drop_unrelated = ['Unnamed: 0', 'Month', 'Name', 'SSN',]

    # Drop columns
    df.drop(columns=columns_to_drop_unrelated, inplace=True)


    # Dropping columns not in used in this model
    ##############################
    columns_to_drop_not_used= ['Type_of_Loan', 'Num_Credit_Inquiries'
                               , 'Amount_invested_monthly'
                               , 'Payment_Behaviour']

    # Drop columns
    df.drop(columns=columns_to_drop_not_used, inplace=True)


    # 1. Cast the column to the correct data type
    # 2. handle missing values
    # 3. handle incorrect values

    # #### Monthly_Balance

    df['Monthly_Balance'] = df['Monthly_Balance'].str.replace('__-333333333333333333333333333__', '')
    df['Monthly_Balance'] = df['Monthly_Balance'].str.replace('_', '')
    df['Monthly_Balance'][(df['Monthly_Balance'] == '')] = None
    df['Monthly_Balance'] = df['Monthly_Balance'].astype(float)
    df['Monthly_Balance'] =  df.groupby('Customer_ID')['Monthly_Balance'].fillna(method='ffill').fillna(method='bfill').astype(float)


    # #### Changed_Credit_Limit

    df['Changed_Credit_Limit'] = df['Changed_Credit_Limit'].str.replace('_', '')
    df['Changed_Credit_Limit'][(df['Changed_Credit_Limit'] == '')] = None
    df['Changed_Credit_Limit'] = df['Changed_Credit_Limit'].astype(float)
    df['Changed_Credit_Limit'] =  df.groupby('Customer_ID')['Changed_Credit_Limit'].fillna(method='ffill').fillna(method='bfill').astype(float)

    # #### Total_EMI_per_month
    df['Total_EMI_per_month'] = df['Total_EMI_per_month'].astype(float)
    df['Total_EMI_per_month'][(df['Total_EMI_per_month'] > 600) | (df['Total_EMI_per_month'] <= 0)] = np.nan 
    df['Total_EMI_per_month'] =  df.groupby('Customer_ID')['Total_EMI_per_month'].fillna(method='ffill').fillna(method='bfill').astype(float)

    # #### Delay_from_due_date
    df['Delay_from_due_date'] = df['Delay_from_due_date'].astype(int)
    df['Delay_from_due_date'] =  df.groupby('Customer_ID')['Delay_from_due_date'].fillna(method='ffill').fillna(method='bfill').astype(int)

    # #### Num_Bank_Accounts
    df['Num_Bank_Accounts'] = df['Num_Bank_Accounts'].astype(int)
    df['Num_Bank_Accounts'][(df['Num_Bank_Accounts'] > 70) | (df['Num_Bank_Accounts'] <= 0)] = np.nan 
    df['Num_Bank_Accounts'] =  df.groupby('Customer_ID')['Num_Bank_Accounts'].fillna(method='ffill').fillna(method='bfill').astype(int)

    # #### Payment_of_Min_Amount
    df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].astype("string")

    # #### Num_of_Loan
    df['Num_of_Loan'] = df['Num_of_Loan'].str.replace('_', '')
    df['Num_of_Loan'] = df['Num_of_Loan'].astype(int)
    df['Num_of_Loan'][(df['Num_of_Loan'] > 100) | (df['Num_of_Loan'] <= 0)] = np.nan 
    df['Num_of_Loan'] =  df.groupby('Customer_ID')['Num_of_Loan'].fillna(method='ffill').fillna(method='bfill').astype(int)

    # #### Credit_History_Age
    df['Credit_History_Age'] = df['Credit_History_Age'].str.replace(r' Years and \d\d? Months', '', regex=True)
    df['Credit_History_Age'] =  df.groupby('Customer_ID')['Credit_History_Age'].fillna(method='ffill').fillna(method='bfill').astype(int)

    # ### 'Num_of_Delayed_Payment'
    df['Num_of_Delayed_Payment'] = df['Num_of_Delayed_Payment'].str.replace('_', '')
    df['Num_of_Delayed_Payment'] =  df.groupby('Customer_ID')['Num_of_Delayed_Payment'].fillna(method='ffill').fillna(method='bfill').astype(int)

    # Cutting off at 18 is the lowest such that 
    # there does not exist a CustomerID where all months are null, therefore
    # 18 is the lowest cut-off that allows back-fill and fore-fill from other months
    df['Num_of_Delayed_Payment'][(df['Num_of_Delayed_Payment'] > 18) | (df['Num_of_Delayed_Payment'] <= 0)] = np.nan 

    df['Num_of_Delayed_Payment'] =  df.groupby('Customer_ID')['Num_of_Delayed_Payment'].fillna(method='ffill').fillna(method='bfill').astype(int)

    df['Num_of_Delayed_Payment'] = df['Num_of_Delayed_Payment'].astype(int)


    # #### Outstanding_Debt

    # Outstanding Debt is all float values, except when it has erroneous underscores at the end.
    # It has no null columns. So with this one command we have nothing but proper float values. 
    df['Outstanding_Debt'] = df['Outstanding_Debt'].str.replace('_', '')

    # The distribution scews left, but there are no crazy outliers
    df['Outstanding_Debt'] = df['Outstanding_Debt'].astype(float)


    # #### Age

    # Extracting non-numeric textual data
    df['Age'][~df['Age'].str.isnumeric()].unique() 

    df['Age'] = df['Age'].str.replace('_', '')

    # get details on column
    df['Age'].describe()


    # In[90]:


    # cast column to integer
    df['Age'] = df['Age'].astype(int)

    # get details on column
    df['Age'].describe()

    # Lets set any inappropriate value which is not at all possible like negative and high positive values above 100 to null for now.
    df['Age'][(df['Age'] > 100) | (df['Age'] <= 0)] = np.nan 

    # get details on column
    df['Age'].describe()


    # if some month's age data is missing then we can simply refer the other months data of same customer to replace with an appropriate value.
    df['Age'] =  df.groupby('Customer_ID')['Age'].fillna(method='ffill').fillna(method='bfill').astype(int)

    # get details on column
    df['Age'].describe()


    # In[93]:


    # Check variable using plots:
    # plot_numerical_column(df, 'Age', 20)


    # #### Occupation

    # In[94]:


    # Extracting unique data
    df['Occupation'].unique()


    # In[95]:


    # There is a placeholder present. We will replace it with null for now and deal with it later.
    df['Occupation'][df['Occupation'] == '_______'] = np.nan

    # get details on column
    df['Occupation'].describe()


    # In[96]:


    # counting null values
    df['Occupation'].isnull().sum()


    # In[97]:


    # So, no one has transitioned into new role or maybe its not mentioned and is coming as null. But these possibilities we obviously can't tell 
    # until and unless we get back to the customer. We need to make the most appropriate educated guess for now.
    df['Occupation'] =  df.groupby('Customer_ID')['Occupation'].fillna(method='ffill').fillna(method='bfill')

    # get details on column
    df['Occupation'].describe()


    # Casting to correct data type
    df['Occupation'] = df['Occupation'].astype("string")

    # ### Annual_Income

    # using regex to find values which don't follow the pattern of a float
    df['Annual_Income'][~df['Annual_Income'].str.fullmatch('([0-9]*[.])?[0-9]+')].unique() 


    # Replacing underscores with empty files
    df['Annual_Income'] = df['Annual_Income'].str.replace('_', '')

    # get details on column
    df['Annual_Income'].describe()


    # Casting to the correct data type
    df['Annual_Income'] = df['Annual_Income'].astype(float)


    # From the data cleaning we know that the Annual Income  max value is 180,000, so we will set all values bigger than 180000 to NA
    df.loc[df['Annual_Income'] > 180000, 'Annual_Income'] = pd.NA
    # df['Annual_Income'][df['Monthly_Inhand_Salary'].notnull()] = df[df['Monthly_Inhand_Salary'].notnull()].groupby(['Customer_ID', 'Monthly_Inhand_Salary'], group_keys = False)['Annual_Income'].transform(return_mode)

    # get details on column
    df['Annual_Income'].describe()


    # In[103]:


    # Since the maximum is to big and probably 
    df['Annual_Income'] = df.groupby('Customer_ID')['Annual_Income'].fillna(method='ffill').fillna(method='bfill')

    # get details on column
    df['Annual_Income'].describe()


    # In[104]:


    # Check variable using plots:
    # plot_numerical_column(df, 'Annual_Income', 30)


    # ### Monthly Inhand Salary

    # In[105]:


    # get details on column
    df['Monthly_Inhand_Salary'].describe()


    # In[106]:


    # Since the maximum is to big and probably 
    df['Monthly_Inhand_Salary'] = df.groupby('Customer_ID')['Monthly_Inhand_Salary'].fillna(method='ffill').fillna(method='bfill')

    # get details on column
    df['Monthly_Inhand_Salary'].describe()


    # In[107]:


    # Check variable using plots:
    # plot_numerical_column(df, 'Monthly_Inhand_Salary', 30)


    # ### Number of Credit Cards

    # In[108]:


    # get details on column
    df['Num_Credit_Card'].describe()


    # In[109]:


    # From the data cleaning we know that the Annual Income  max value is 180,000, so we will set all values bigger than 10 to NA
    df.loc[df['Num_Credit_Card'] > 11, 'Num_Credit_Card'] = pd.NA

    # get details on column
    df['Num_Credit_Card'].describe()


    # In[110]:


    # Since the maximum is to big and probably 
    df['Num_Credit_Card'] = df.groupby('Customer_ID')['Num_Credit_Card'].fillna(method='ffill').fillna(method='bfill')

    # get details on column
    df['Num_Credit_Card'].describe()


    # In[111]:


    # Check variable using plots:
    # plot_numerical_column(df, 'Num_Credit_Card', 30)


    # ### Interest Rate

    # get details on column
    df['Interest_Rate'].describe()


    # In[113]:


    # From the data cleaning we know that the Annual Income  max value is 180,000, so we will set all values bigger than 10 to NA
    df.loc[df['Interest_Rate'] > 34, 'Interest_Rate'] = pd.NA

    # get details on column
    df['Interest_Rate'].describe()


    # In[114]:


    # filling NA with median
    df['Interest_Rate'] = df.groupby('Customer_ID')['Interest_Rate'].transform(lambda x: x.median())

    # get details on column
    df['Interest_Rate'].describe()


    # ### Credit Mix

    # In[115]:


    # get details on column
    df['Credit_Mix'].describe()


    # In[116]:


    # The column contains underscores as placeholders. We will replace them with null for now.
    df['Credit_Mix'][df['Credit_Mix'] == '_'] = np.nan


    # In[117]:


    # From the data clean file, we will fill forward and backward any missing data
    df['Credit_Mix'] = df.groupby('Customer_ID')['Credit_Mix'].fillna(method='ffill').fillna(method='bfill')

    # get details on column
    df['Credit_Mix'].describe()


    # In[118]:


    # Casting to correct data type
    df['Credit_Mix'] = df['Credit_Mix'].astype("string")

    # Check variable using plots:
    # plot_no_numerical_column(df, 'Credit_Mix')


    # ### Credit Utilization Ratio

    # In[119]:


    # get details on column
    df['Credit_Utilization_Ratio'].describe()

    # ### Credit Score

    # Casting to correct data type
    df['Credit_Score'] = df['Credit_Score'].astype("string")

    # ### Check for duplicates

    # This is only checking for exact duplicates. Are there duplicates that we are missing?
    #HINT What happens if a fire exists across multiple counties.
    df.duplicated().sum()


    # ### Look at summary data

    df.describe()

    df.info()

    # We have selected some features below, you can add more.
    #HINT: look at the dtypes above
    #    l_target = ['Credit_Score']
    #    l_cols_numerical = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary'
    #                        , 'Num_Credit_Card', 'Interest_Rate'
    #                        , 'Credit_Utilization_Ratio', 'Credit_History_Age'
    #                        , 'Num_of_Delayed_Payment'
    #                        ] 
    #    l_cols_categorical = ['Occupation', 'Credit_Mix']
    # 
    # 
    #    # ### Visualize variables
    # 
    #    # Look at Histograms
    #    df[l_cols_numerical].hist(xlabelsize =6)
    #    # plt.tight_layout()
    # 
    #    # Look at scattered plots
    #    axs = pd.plotting.scatter_matrix(df[l_cols_numerical], figsize=(10,10), marker = 'o', hist_kwds = {'bins': 10}, s = 60, alpha = 0.2)
    # 
    #    def wrap(txt, width=8):
    #        '''helper function to wrap text for long labels'''
    #        import textwrap
    #        return '\n'.join(textwrap.wrap(txt, width))
    # 
    #    for ax in axs[:,0]: # the left boundary
    #        ax.set_ylabel(wrap(ax.get_ylabel()), size = 8)
    #        ax.set_xlim([None, None])
    #        ax.set_ylim([None, None])
    # 
    #    for ax in axs[-1,:]: # the lower boundary
    #        ax.set_xlabel(wrap(ax.get_xlabel()), size = 8)
    #        ax.set_xlim([None, None])
    #        ax.set_ylim([None, None])
    # 
    # 
    #    # Box plots for all variables 
    #    df[l_cols_numerical].boxplot(rot=90)

    # ### Feature Derivation/Engineering

    # ## Feature Selection

    # Drop costumer ID and cast ID
    df = df.drop(columns='Customer_ID')

    df['ID'] = df['ID'].astype('string')

    # Getting information on the data frame
    df.info()

    # If you want to change the variables for your model, do that here!
    target = ['Credit_Score']
    continuous_features = ['Age', 'Annual_Income', 'Monthly_Balance'
                           , 'Monthly_Inhand_Salary', 'Num_Credit_Card'
                           , 'Interest_Rate', 'Credit_Utilization_Ratio'
                           , 'Credit_History_Age', 'Outstanding_Debt'
                           , 'Num_of_Loan', 'Num_of_Delayed_Payment'
                           , 'Num_Bank_Accounts', 'Delay_from_due_date'
                           , 'Total_EMI_per_month', 'Changed_Credit_Limit'
                           ] 
    categorical_features = ['Occupation', 'Credit_Mix', 'Payment_of_Min_Amount']


    # Encode variables to use in Neural Network
    # A one hot encoding is appropriate for categorical data where no relationship exists between categories.
    # It involves representing each categorical variable with a binary vector that has one element for each 
    # unique label and marking the class label with a 1 and all other elements 0.

    # For example, if our variable was “color” and the labels were “red,” “green,” and “blue,” we would encode 
    # each of these labels as a three-element binary vector as follows:

    # Red: [1, 0, 0]
    # Green: [0, 1, 0]
    # Blue: [0, 0, 1]

    # Then each label in the dataset would be replaced with a vector (one column becomes three). 
    # This is done for all categorical variables so that our nine input variables or columns become 
    # 43 in the case of the breast cancer dataset.

    # The scikit-learn library provides the OneHotEncoder to automatically one hot encode one or more variables.

    # Encoder for input features
    encoder = OneHotEncoder(handle_unknown='ignore')

    # Encoder for target
    le = LabelEncoder()

    # Scaler for continuous input features
    scaler = MinMaxScaler()

    ## CATEGORICAL INPUT FEATURES
    # Encoding categorical features
    encoded_features = encoder.fit_transform(df[categorical_features])

    # Convert the encoded data back to a DataFrame:
    encoded_df = pd.DataFrame(encoded_features.toarray(), columns=encoder.get_feature_names_out(categorical_features))

    # joining dataframes 
    df = pd.concat([df, encoded_df], axis=1)
    print(df.info())


    ## TARGET FEATURES
    # Encoding categorical features
    encoded_target = encoder.fit_transform(df[target])

    # Convert the encoded data back to a DataFrame:
    encoded_target_df = pd.DataFrame(encoded_target.toarray(), columns=encoder.get_feature_names_out(target))

    # joining dataframes 
    df = pd.concat([df, encoded_target_df], axis=1)
    print(df.info())


    ## INPUT CONTINUOUS FEATURES
    # Scaling continuous features
    scaled_features = scaler.fit_transform(df[continuous_features])

    # Convert the scaled data back to a DataFrame:
    encoded_scaled_df = pd.DataFrame(scaled_features, columns=scaler.get_feature_names_out(continuous_features))

    # joining dataframes 
    df.drop(columns=continuous_features, inplace=True)
    df = pd.concat([df, encoded_scaled_df], axis=1)
    print(df.info())


    # ## Modeling

    # Constructing dataframe for modeling
    features_for_model = ['Age', 'Annual_Income', 'Monthly_Balance'
                          , 'Monthly_Inhand_Salary'
                          , 'Num_Credit_Card', 'Credit_History_Age', 'Outstanding_Debt'
                          , 'Interest_Rate', 'Credit_Utilization_Ratio'
                          , 'Num_of_Loan', 'Num_of_Delayed_Payment'
                          , 'Num_Bank_Accounts', 'Delay_from_due_date'
                          , 'Total_EMI_per_month', 'Changed_Credit_Limit'

                          , 'Credit_Mix_Bad', 'Credit_Mix_Good', 'Credit_Mix_Standard'

                          , 'Payment_of_Min_Amount_NM', 'Payment_of_Min_Amount_No'
                          , 'Payment_of_Min_Amount_Yes'

                           , 'Occupation_Accountant', 'Occupation_Architect'
                           , 'Occupation_Developer', 'Occupation_Doctor'
                           , 'Occupation_Engineer', 'Occupation_Entrepreneur'
                           , 'Occupation_Journalist', 'Occupation_Lawyer'
                           , 'Occupation_Manager', 'Occupation_Mechanic'
                           ,'Occupation_Media_Manager', 'Occupation_Musician'
                           , 'Occupation_Scientist', 'Occupation_Teacher'
                           , 'Occupation_Writer'
                          ] 

    target_features = ['Credit_Score_Good', 'Credit_Score_Poor', 'Credit_Score_Standard']

    # Getting the size of input size
    print(len(features_for_model))





def main_menu():
    while True:
        print("\nMain Menu")
        print("(1) Load data")
        print("(2) Process data")
        print("(3) Model details")
        print("(4) Test model")
        print("(5) Quit")
        
        choice = input("Enter your choice (1-5): ")
        
        if choice == '1':
            print("\nLoading and cleaning input data set")
            print("***********************************")

        elif choice == '2':
            print("\nProcessing input data set")
            print("*************************")

        elif choice == '3':
            print("\nPrinting model details")
            print("**********************")

        elif choice == '4':
            print("\nTesting model")
            print("*************")

        elif choice == '5':
            print("\nQuiting program, goodbye!")

            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()


