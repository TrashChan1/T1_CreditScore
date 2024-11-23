
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
columns_to_drop_not_used= ['Num_Bank_Accounts', 'Num_of_Loan', 'Type_of_Loan', 'Delay_from_due_date', 'Num_of_Delayed_Payment'
                           , 'Changed_Credit_Limit', 'Payment_of_Min_Amount', 'Num_Credit_Inquiries'
                           , 'Payment_of_Min_Amount', 'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance', 'Payment_Behaviour']

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
# 3. Plot numerical columns to make sure distributions are correct


# #### Credit_History_Age

# I don't think that the months of the age matters, so we can just extract the number at the beginning of the string
df['Credit_History_Age'] = df['Credit_History_Age'].str.replace(r' Years and \d\d? Months', '', regex=True)



# This still leaves us with 7245 null values. Let's try backfill.
df['Credit_History_Age'] =  df.groupby('Customer_ID')['Credit_History_Age'].fillna(method='ffill').fillna(method='bfill').astype(int)

# Backfill worked, we have no null values. Also, the distribution is very normal. The column should now be nice and clean.


# #### Outstanding_Debt

# Outstanding Debt is all float values, except when it has erroneous underscores at the end.
# It has no null columns. So with this one command we have nothing but proper float values. 
df['Outstanding_Debt'] = df['Outstanding_Debt'].str.replace('_', '')

# The distribution scews left, but there are no crazy outliers
df['Outstanding_Debt'] = df['Outstanding_Debt'].astype(float)


# #### Age

# Extracting non-numeric textual data
df['Age'][~df['Age'].str.isnumeric()].unique() 


# In[89]:


# Looking at the above values, looks like many underscores are present in our dataset. 
# For age, they are not needed and can be replaced with blanks. 
# Some negative values are also present which we will handle later on.

df['Age'] = df['Age'].str.replace('_', '')

# get details on column
df['Age'].describe()


# In[90]:


# cast column to integer
df['Age'] = df['Age'].astype(int)

# get details on column
df['Age'].describe()


# In[91]:


# As already noted in the data cleaning notebook,  there are many extreme values present in age column which are unrealistic in nature. 
# Lets set any inappropriate value which is not at all possible like negative and high positive values above 100 to null for now.
df['Age'][(df['Age'] > 100) | (df['Age'] <= 0)] = np.nan 

# get details on column
df['Age'].describe()


# In[92]:


# We have removed all inappropriate values and replaced them with nulls. What we have in our dataset is customer data, 
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


# In[98]:


# Casting to correct data type
df['Occupation'] = df['Occupation'].astype("string")

# Check variable using plots:
# plot_no_numerical_column(df, 'Occupation')


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
l_target = ['Credit_Score']
l_cols_numerical = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Credit_Card', 'Interest_Rate', 'Credit_Utilization_Ratio', 'Credit_History_Age'] 
l_cols_categorical = ['Occupation', 'Credit_Mix']


# ### Visualize variables

# Look at Histograms
df[l_cols_numerical].hist(xlabelsize =6)
# plt.tight_layout()

# Look at scattered plots
axs = pd.plotting.scatter_matrix(df[l_cols_numerical], figsize=(10,10), marker = 'o', hist_kwds = {'bins': 10}, s = 60, alpha = 0.2)

def wrap(txt, width=8):
    '''helper function to wrap text for long labels'''
    import textwrap
    return '\n'.join(textwrap.wrap(txt, width))

for ax in axs[:,0]: # the left boundary
    ax.set_ylabel(wrap(ax.get_ylabel()), size = 8)
    ax.set_xlim([None, None])
    ax.set_ylim([None, None])

for ax in axs[-1,:]: # the lower boundary
    ax.set_xlabel(wrap(ax.get_xlabel()), size = 8)
    ax.set_xlim([None, None])
    ax.set_ylim([None, None])


# Box plots for all variables 
df[l_cols_numerical].boxplot(rot=90)

# ### Feature Derivation/Engineering

# ## Feature Selection

# Drop costumer ID and cast ID
df = df.drop(columns='Customer_ID')

df['ID'] = df['ID'].astype('string')

# Getting information on the data frame
df.info()

# If you want to change the variables for your model, do that here!
target = ['Credit_Score']
continuous_features = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Credit_Card', 'Interest_Rate', 'Credit_Utilization_Ratio'] 
categorical_features = ['Occupation', 'Credit_Mix']


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


# Encoding categorical features
encoded_features = encoder.fit_transform(df[categorical_features])

# Convert the encoded data back to a DataFrame:
encoded_df = pd.DataFrame(encoded_features.toarray(), columns=encoder.get_feature_names_out(categorical_features))

# joining dataframes 
df = pd.concat([df, encoded_df], axis=1)
print(df.info())

# Encoding categorical features
encoded_target = encoder.fit_transform(df[target])

# Convert the encoded data back to a DataFrame:
encoded_target_df = pd.DataFrame(encoded_target.toarray(), columns=encoder.get_feature_names_out(target))

# joining dataframes 
df = pd.concat([df, encoded_target_df], axis=1)
print(df.info())

