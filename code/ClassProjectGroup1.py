
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


def clean_data(df: pd.core.frame.DataFrame):

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
                               , 'Payment_Behaviour'
                               ]

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

    df['Num_of_Delayed_Payment'][(df['Num_of_Delayed_Payment'] > 18) | (df['Num_of_Delayed_Payment'] <= 0)] = np.nan 

    df['Num_of_Delayed_Payment'] =  df.groupby('Customer_ID')['Num_of_Delayed_Payment'].fillna(method='ffill').fillna(method='bfill').astype(int)

    df['Num_of_Delayed_Payment'] = df['Num_of_Delayed_Payment'].astype(int)


    # #### Outstanding_Debt

    df['Outstanding_Debt'] = df['Outstanding_Debt'].str.replace('_', '')
    df['Outstanding_Debt'] = df['Outstanding_Debt'].astype(float)

    # #### Age

    # Extracting non-numeric textual data
    df['Age'][~df['Age'].str.isnumeric()].unique() 
    df['Age'] = df['Age'].str.replace('_', '')

    # cast column to integer
    df['Age'] = df['Age'].astype(int)

    # Lets set any inappropriate value which is not at all possible like negative and high positive values above 100 to null for now.
    df['Age'][(df['Age'] > 100) | (df['Age'] <= 0)] = np.nan 

    df['Age'] =  df.groupby('Customer_ID')['Age'].fillna(method='ffill').fillna(method='bfill').astype(int)

    # #### Occupation

    df['Occupation'][df['Occupation'] == '_______'] = np.nan
    df['Occupation'] =  df.groupby('Customer_ID')['Occupation'].fillna(method='ffill').fillna(method='bfill')
    df['Occupation'] = df['Occupation'].astype("string")

    # ### Annual_Income
    df['Annual_Income'][~df['Annual_Income'].str.fullmatch('([0-9]*[.])?[0-9]+')].unique() 
    df['Annual_Income'] = df['Annual_Income'].str.replace('_', '')
    df['Annual_Income'] = df['Annual_Income'].astype(float)
    df.loc[df['Annual_Income'] > 180000, 'Annual_Income'] = pd.NA
    df['Annual_Income'] = df.groupby('Customer_ID')['Annual_Income'].fillna(method='ffill').fillna(method='bfill')


    # ### Monthly Inhand Salary
    df['Monthly_Inhand_Salary'] = df.groupby('Customer_ID')['Monthly_Inhand_Salary'].fillna(method='ffill').fillna(method='bfill')

    # ### Number of Credit Cards
    df.loc[df['Num_Credit_Card'] > 11, 'Num_Credit_Card'] = pd.NA
    df['Num_Credit_Card'] = df.groupby('Customer_ID')['Num_Credit_Card'].fillna(method='ffill').fillna(method='bfill')

    # ### Interest Rate
    df.loc[df['Interest_Rate'] > 34, 'Interest_Rate'] = pd.NA
    df['Interest_Rate'] = df.groupby('Customer_ID')['Interest_Rate'].transform(lambda x: x.median())

    # ### Credit Mix
    df['Credit_Mix'][df['Credit_Mix'] == '_'] = np.nan
    df['Credit_Mix'] = df.groupby('Customer_ID')['Credit_Mix'].fillna(method='ffill').fillna(method='bfill')
    df['Credit_Mix'] = df['Credit_Mix'].astype("string")

    # ### Credit Utilization Ratio
    # Surprisingly, fine as is.

    # ### Credit Score
    df['Credit_Score'] = df['Credit_Score'].astype("string")

    # Drop costumer ID and cast ID
    df = df.drop(columns='Customer_ID')

    df['ID'] = df['ID'].astype('string')

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

    # Encoder for input features
    encoder = OneHotEncoder(handle_unknown='ignore')

    # Scaler for continuous input features
    scaler = MinMaxScaler()

    ## CATEGORICAL INPUT FEATURES
    # Encoding categorical features
    encoded_features = encoder.fit_transform(df[categorical_features])

    # Convert the encoded data back to a DataFrame:
    encoded_df = pd.DataFrame(encoded_features.toarray(), columns=encoder.get_feature_names_out(categorical_features))

    # joining dataframes 
    df = pd.concat([df, encoded_df], axis=1)

    ## TARGET FEATURES
    # Encoding categorical features
    encoded_target = encoder.fit_transform(df[target])

    # Convert the encoded data back to a DataFrame:
    encoded_target_df = pd.DataFrame(encoded_target.toarray(), columns=encoder.get_feature_names_out(target))

    # joining dataframes 
    df = pd.concat([df, encoded_target_df], axis=1)

    ## INPUT CONTINUOUS FEATURES
    # Scaling continuous features
    scaled_features = scaler.fit_transform(df[continuous_features])

    # Convert the scaled data back to a DataFrame:
    encoded_scaled_df = pd.DataFrame(scaled_features, columns=scaler.get_feature_names_out(continuous_features))

    # joining dataframes 
    df.drop(columns=continuous_features, inplace=True)
    df = pd.concat([df, encoded_scaled_df], axis=1)

    return df

def construct_model(df: pd.core.frame.DataFrame):
    dropout = 0.2
    noise = 0.05

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

    # Defining data sets
    X = df[features_for_model].to_numpy()
    y = df[target_features].to_numpy()

    # Basic train-test split
    # 80% training and 20% test 
    X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.20, random_state=42)

    # ### Neural Network

    # Set up the layers
    ###################

    # Create network topology
    model = keras.Sequential()

    model.add(Dense(256, input_dim = X_train.shape[1], activation = 'relu'))

    # Noise Helps fit data rather than noise
    model.add(keras.layers.GaussianNoise(noise))

    model.add(keras.layers.Dense(256, activation="relu"))

    # dropout helps fit data rather than noise.
    # By applyinng ot on the layer that goes from many neurons to few neurons makes it effective without ruining the brains of the model
    model.add(keras.layers.Dropout(dropout))

    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dense(12, activation="relu"))

    # output layer
    # For classification tasks, we generally tend to add an activation function in the output ("sigmoid" for binary, and "softmax" for multi-class, etc.).
    model.add(keras.layers.Dense(3, activation="softmax"))

    print(model.summary())


    # Compile the Model
    ###################
    # Before the model is ready for training, it needs a few more settings. These are added during the model's compile step:
    # Loss function —This measures how accurate the model is during training. You want to minimize this function to "steer" the model in the right direction.
    # Optimizer —This is how the model is updated based on the data it sees and its loss function.
    # Metrics —Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified.

    # compile the model


    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])


    # Train the Model
    #################
    # Training the neural network model requires the following steps:
    # Feed the training data to the model. In this example, the training data is in the train_images and train_labels arrays.
    # To start training, call the model.fit method—so called because it "fits" the model to the training data:

    # callback stops the epochs early when reaching plateu
    callback = keras.callbacks.EarlyStopping(monitor='loss',
                                                  patience=3)

    model.fit(X_train, y_train, epochs = 256, batch_size = 512, callbacks=[callback])

    return model, X_test, y_test

def test_model(model: keras.models.Sequential, X_test: np.ndarray, y_test: np.ndarray, df: pd.core.frame.DataFrame):

    encoder = OneHotEncoder(handle_unknown='ignore')
    #encoder.fit_transform(df[categorical_features])

    #Evaluate accuracy
    test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
    print('\nTest accuracy:', test_acc)
    print('\nLoss:', test_loss)


    # ## Make Predictions

    # Make Predictions

    # Here's the predictions. I guess these need exported to a csv file, but IDK what for
    predictions = model.predict(X_test)

    # Here, the model has predicted the label for each image in the testing set. Let's take a look at some predictions
    print(predictions[0])
    print(predictions[10])
    print(predictions[100])
    print(predictions[1000])
    print(predictions[10000])


    # 3 different credit scores. You can see the comparison between the trained and tested values
    
    # The following functionality is broken and needs fixed:
'
    # getting y_test values
    y_tested = encoder.inverse_transform(y_test)


    # getting the value of the predictions
    y_predicted = encoder.inverse_transform(predictions)

    # printing the first 15 values of the test and predicted values 
    data = []
    for i in range(15):
        data.append([y_tested[i], y_predicted[i]])

    headers = ["True Value", "Predicted Value"]

    print(tabulate(data, headers=headers, tablefmt="grid"))
'



def main_menu():
    df = 0
    while True:
        print("\nMain Menu")
        print("(1) Load data")
        print("(2) Process data")
        print("(3) Model details")
        print("(4) Test model")
        print("(5) Quit")
        
        choice = input("Enter your choice (1-5): ")
        
        if choice == '1':
            df = handle_load_data()
            df.info()
            #if(df.empty()):
                #continue

        elif choice == '2':
            print("\nProcessing input data set")
            print("*************************")
            df = clean_data(df)
            df.info()

        elif choice == '3':
            print("\n Compiling the model")
            print("**********************")
            model, X_test, y_test = construct_model(df)

        elif choice == '4':
            print("\nTesting model")
            print("*************")
            test_model(model, X_test, y_test, df)

        elif choice == '5':
            print("\nQuiting program, goodbye!")

            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()


