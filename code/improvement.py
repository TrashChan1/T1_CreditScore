#!/usr/bin/env python

# General Packages
import math
import os
from pathlib import Path

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


# Get the current working directory
current_dir = os.getcwd() 

# Construct a path to the parent directory
parent_dir = os.path.dirname(current_dir)

# Access a file in the parent directory
file_path = os.path.join(parent_dir, "data")
file_path = os.path.join(file_path, "credit_score_data.csv")

# Load Credit Score data
df = pd.read_csv(file_path) 


# ## Performing Basic Data Cleaning

# ### Check for missing variables

# This will count the number of missing values per column, we will handle tis at each column


df.isna().sum()


# ### Dropping some columns
# From CS3500_Credit_score_classification_data_cleaning.ipynb we determined that some columns are not necessary. Also, for this model we will only consider an small amount of variables

# In[87]:


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

# checking Columns
df.info()


# ### Analyze each column and model
# 
# We will perform the following tasks for each columns in the model:
# 
# 1. Cast the column to the correct data type
# 2. handle missing values
# 3. handle incorrect values


# #### Monthly_Balance

df['Monthly_Balance'] = df['Monthly_Balance'].str.replace('__-333333333333333333333333333__', '')
df['Monthly_Balance'] = df['Monthly_Balance'].str.replace('_', '')
df['Monthly_Balance'][(df['Monthly_Balance'] == '')] = None
df['Monthly_Balance'] = df['Monthly_Balance'].astype(float)
# df['Monthly_Balance'][(df['Monthly_Balance'] > 1000) | (df['Monthly_Balance'] <= 0)] = np.nan 
df['Monthly_Balance'] =  df.groupby('Customer_ID')['Monthly_Balance'].fillna(method='ffill').fillna(method='bfill').astype(float)


# #### Changed_Credit_Limit

df['Changed_Credit_Limit'] = df['Changed_Credit_Limit'].str.replace('_', '')
df['Changed_Credit_Limit'][(df['Changed_Credit_Limit'] == '')] = None

df['Changed_Credit_Limit'] = df['Changed_Credit_Limit'].astype(float)

# df['Changed_Credit_Limit'][(df['Changed_Credit_Limit'] > 16) | (df['Changed_Credit_Limit'] <= 0)] = np.nan 

df['Changed_Credit_Limit'] =  df.groupby('Customer_ID')['Changed_Credit_Limit'].fillna(method='ffill').fillna(method='bfill').astype(float)

# #### Total_EMI_per_month

df['Total_EMI_per_month'] = df['Total_EMI_per_month'].astype(float)

df['Total_EMI_per_month'][(df['Total_EMI_per_month'] > 600) | (df['Total_EMI_per_month'] <= 0)] = np.nan 

df['Total_EMI_per_month'] =  df.groupby('Customer_ID')['Total_EMI_per_month'].fillna(method='ffill').fillna(method='bfill').astype(float)


# #### Delay_from_due_date

df['Delay_from_due_date'] = df['Delay_from_due_date'].astype(int)

# df['Delay_from_due_date'][(df['Delay_from_due_date'] > 40) | (df['Delay_from_due_date'] <= 0)] = np.nan 

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

# I don't think that the months of the age matters, so we can just extract the number at the beginning of the string
df['Credit_History_Age'] = df['Credit_History_Age'].str.replace(r' Years and \d\d? Months', '', regex=True)

# This still leaves us with 7245 null values. Let's try backfill.
df['Credit_History_Age'] =  df.groupby('Customer_ID')['Credit_History_Age'].fillna(method='ffill').fillna(method='bfill').astype(int)

# Backfill worked, we have no null values. Also, the distribution is very normal. The column should now be nice and clean.

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


# Defining data sets
X = df[features_for_model].to_numpy()
y = df[target_features].to_numpy()
y = df2[target_features].to_numpy()
print(y)


# ### Train / test split

# In[140]:


# Basic train-test split
# 80% training and 20% test 
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.20, random_state=42)

# Checking the dimensions of the variables
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[141]:


# Printing X_train and y_train
print(X_train)
print(y_train)


# ### Neural Network

# In[142]:


# Set up the layers
###################
# The basic building block of a neural network is the layer. Layers extract representations from the data fed into them. Hopefully, these representations are meaningful for the problem at hand.
# Most of deep learning consists of chaining together simple layers. Most layers, such as tf.keras.layers.Dense, have parameters that are learned during training.

# Create network topology
model = keras.Sequential()

# Adding input model --> 24 input layers
#model.add(keras.layers.UnitNormalization())
model.add(Dense(256, input_dim = X_train.shape[1], activation = 'relu'))

dropout = 0.2

# Adding hidden layers
model.add(keras.layers.GaussianNoise(0.05))
model.add(keras.layers.Dense(256, activation="relu"))
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


# Train the Model
#################
# Training the neural network model requires the following steps:
# Feed the training data to the model. In this example, the training data is in the train_images and train_labels arrays.
# To start training, call the model.fit method—so called because it "fits" the model to the training data:

callback = keras.callbacks.EarlyStopping(monitor='loss',
                                              patience=3)

model.fit(X_train, y_train, epochs = 256, batch_size = 512, callbacks=[callback])


# In[145]:


#Evaluate accuracy
test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
print('\nTest accuracy:', test_acc)
print('\nLoss:', test_loss)


# ## Make Predictions

# Make Predictions
predictions = model.predict(X_test)

# Here, the model has predicted the label for each image in the testing set. Let's take a look at some predictions
print(predictions[0])
print(predictions[10])
print(predictions[100])
print(predictions[1000])
print(predictions[10000])


# 3 different credit scores. You can see the comparison between the trained and tested values

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


# In[ ]:


# Confusion Matrix
##################

# A confusion matrix for 3 variables is a table that visually represents how well a classification model performs when 
# predicting three different categories, where each row represents the actual class and each column represents the predicted class,
# resulting in a 3x3 grid that shows how many instances were correctly classified and how many were misclassified between each 
# of the three possible categories; essentially, it provides a detailed breakdown of the model's errors for each class in 
# a multi-class classification problem.

# Key points about a 3-variable confusion matrix:
################################################
# Structure:
# The matrix has 3 rows and 3 columns, where each row represents one of the actual classes and each column represents one of the predicted classes. 
 
# Diagonal elements:
# The diagonal cells of the matrix represent the correctly classified instances for each class. 
 
# Off-diagonal elements:
# The values in off-diagonal cells represent the misclassified instances, showing which class the model tends to confuse with another. 

# Class labels
class_labels=['Good', 'Poor', 'Standard']

# plot_prediction_vs_test_categorical(y_tested, y_predicted, class_labels)


# In[149]:


# Explanation of Metrics
########################

# Accuracy: The proportion of correctly classified samples.
# Precision: The ability of the classifier not to label a negative sample as positive.
# Recall: The ability of the classifier to find all the positive samples.
# F1-score: A weighted average of precision and recall.
# Confusion Matrix: A table showing the number of true positives, true negatives, false positives, and false negatives for each class. 
 
# Important Considerations:
# Averaging:
# The average parameter in precision_score, recall_score, and f1_score can be set to different values:
# 'macro': Calculates the metric for each label, and finds their unweighted mean.
# 'micro': Calculates the metric globally by counting the total true positives, false negatives, and false positives.
# 'weighted': Calculates the metric for each label, and finds their average weighted by support (the number of true instances for each label).
# Class Imbalances:
# If your dataset has class imbalances, consider using metrics like f1_score and recall that are less sensitive to this issue.

# Calculating perofrmace of model
calculate_performance_multiclass(y_tested, y_predicted)

