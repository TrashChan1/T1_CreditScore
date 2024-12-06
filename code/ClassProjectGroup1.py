
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
    print("\nLoading and cleaning input data set:")
    print("************************************")
    print(f"[{get_current_time()}] Loading training data set")
    
    # Start timing
    start_time = time.time()
    
    # Load the data
    current_dir = Path.cwd()
    data_folder = current_dir.parent / folder_name
    file_path = data_folder / file_name
    
    # Check if file exists
    if not file_path.exists():
        raise Exception(f"File not found at '{file_path}'")
        
    # Check file size
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
    if file_size > 1000:  # Example: 1GB limit
        raise Exception("File size ({file_size:.2f}MB) exceeds maximum allowed size")
        
    # Check permissions
    if not os.access(file_path, os.R_OK):
        raise Exception("Permission denied - Cannot read file")
        
    # Check if file is empty
    if file_size == 0:
        raise Exception("File is empty")
        
    # Detect file encoding
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
    
    # Try to load the CSV file
    df = pd.read_csv(file_path, encoding=encoding)

    
    if df.empty:
        raise Exception("No data found in file")
        
    # Basic format validation
    if len(df.columns) < 2:  # Assuming we expect at least 2 columns
        raise Exception("Invalid file format - Incorrect number of columns")
    
    # Calculate loading time
    load_time = time.time() - start_time
    
    # Print the required information
    print(f"[{get_current_time()}] Total Columns Read: {len(df.columns)}")
    print(f"[{get_current_time()}] Total Rows Read: {len(df)}")
    print(f"\nTime to load is: {load_time:.2f} seconds")
    
    return df, load_time

def handle_load_data():
    """
    Handle the load data menu option with retry capabilities.
    
    Returns:
        pd.DataFrame or None: The loaded DataFrame if successful, None if user chooses to return to menu
    """
    
    new_file = input("Enter the name of the file to load: ")

    while True:

        # Default behavior scipped by else statement
        try:
            df, load_time = load_data(new_file)
            #df.info()
            try:
                df, encoder = clean_data(df)
                return df, encoder
            except Exception as e:
                print("\nFailed to clean file: ", e)
        except Exception as e:
            print("\nFailed to load file: ", e)

        print("\nWhat would you like to do?")
        print("(1) Try loading the same file again")
        print("(2) Try loading a different file")
        print("(3) Return to main menu")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            pass
        elif choice == '2':
            new_file = input("Enter the name of the file to load: ")
        elif choice == '3':
            raise Exception("File not loaded")
        else:
            print("Invalid choice. Please try again.")
            continue
def clean_data(df: pd.core.frame.DataFrame):

    print(f"[{get_current_time()}] Cleaning data set")
    start_time = time.time()

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

    balance_frame = pd.to_numeric(df.Monthly_Balance, errors='coerce').dropna().to_frame()
    df.drop(columns='Monthly_Balance', inplace=True)
    df = pd.concat([df, balance_frame], axis=1)
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

    print(f"[{get_current_time()}] Rows after cleaning: {len(df)}")
    load_time = time.time() - start_time
    print(f"\nTime to clean is: {load_time:.2f} seconds")
    return df, encoder

def construct_model(df: pd.core.frame.DataFrame, X_train: np.ndarray, y_train: np.ndarray):
    print(f"[{get_current_time()}] Constructing NN")
    start_time = time.time()

    dropout = 0.2
    noise = 0.03

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

    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(128, activation="relu"))
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
    
    load_time = time.time() - start_time
    print(f"\nTime to construct: {load_time:.2f} seconds")
    start_time = time.time()
    print(f"[{get_current_time()}] Training NN")

    # Train the Model
    #################
    # Training the neural network model requires the following steps:
    # Feed the training data to the model. In this example, the training data is in the train_images and train_labels arrays.
    # To start training, call the model.fit method—so called because it "fits" the model to the training data:

    # callback stops the epochs early when reaching plateu
    callback = keras.callbacks.EarlyStopping(monitor='loss',
                                                  patience=3)

    model.fit(X_train, y_train, epochs = 256, batch_size = 512, verbose=0, callbacks=[callback])

    load_time = time.time() - start_time
    print(f"\nTime to train: {load_time:.2f} seconds")

    return model

def handle_train_test_split(df: pd.core.frame.DataFrame, test_size: float = 0.20):
    
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
    return X_train, X_test, y_train, y_test


def calculate_performance_multiclass(y_true, y_pred):
    # Calculates various performance metrics for multiclass classification.

    # Args:
    #     y_true: The true labels.
    #     y_pred: The predicted labels.

    # Returns:
    #     A dictionary containing the calculated metrics.

    metrics = {}

    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)

    # Precision, Recall, and F1-score (macro-averaged)
    metrics['precision'] = precision_score(y_true, y_pred, average='macro')
    metrics['recall'] = recall_score(y_true, y_pred, average='macro')
    metrics['f1_score'] = f1_score(y_true, y_pred, average='macro')

    # Confusion Matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)

    return metrics



def test_model(y_test: np.ndarray, X_test: np.ndarray, model: keras.models.Sequential, encoder: OneHotEncoder):

    print(f"[{get_current_time()}] Testing Model based on 20% of most recently loaded file.")
    start_time = time.time()

    class_labels=['Good', 'Poor', 'Standard']

    # the true credit_score values from the train_test split
    y_tested = encoder.inverse_transform(y_test)

    predictions = model.predict(X_test)
    y_pred = encoder.inverse_transform(predictions)
    predDF = pd.DataFrame(y_pred)
    predDF.to_csv("predictionClassProject1.csv")

    #Evaluate accuracy
    test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
    metrics = calculate_performance_multiclass(y_tested, y_pred) 
    test_acc = metrics['accuracy']
    prec = metrics['precision']
    f1 = metrics['f1_score']

    print(f"[{get_current_time()}] Test accuracy:", test_acc)
    print(f"[{get_current_time()}] Loss:", test_loss)
    print(f"[{get_current_time()}] precision:", prec)
    print(f"[{get_current_time()}] f1_score:", f1)

    # printing the first 15 values of the test and predicted values 
    data = []
    for i in range(15):
        data.append([y_tested[i], y_pred[i]])

    headers = ["True Value", "Predicted Value"]

    print(tabulate(data, headers=headers, tablefmt="grid"))

    load_time = time.time() - start_time
    print(f"\nTime to test: {load_time:.2f} seconds")

    #TODO: Confusion matrix.

    # Plot the confusion matrix
    cm = metrics['confusion_matrix']
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

def main_menu():
    while True:
        print("\nMain Menu")
        print("(1) Load data")
        print("(2) Train NN")
        print("(3) Test model")
        print("(4) Quit")
        
        choice = input("Enter your choice (1-4): ")

        if choice == '1':
            try: 
                df, encoder = handle_load_data()
                #df.info()
                X_train, X_test, y_train, y_test = handle_train_test_split(df, 0.20)
                # print("training based on random 80% of loaded file, and testing on other 20%")
            except Exception as e:
                print("file not loaded: ", e, "\n")

        elif choice == '2':
            print("\nTraining Nueral Network based on input file")
            print("*********************************************")
            try:
                model = construct_model(df, X_train, y_train)
            except Exception as e:
                print("Model not constructed: ", e, "\n")

        elif choice == '3':
            print("\nTesting model")
            print("*************")
            try:
                test_model(y_test, X_test, model, encoder)
            except Exception as e:
                print("Failed to test model: ", e, "\n")

        elif choice == '4':
            print("\nQuiting program, goodbye!")
            break

        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()


