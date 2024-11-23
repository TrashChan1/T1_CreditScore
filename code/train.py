
# ## Modeling

# In[138]:


# Constructing dataframe for modeling
features_for_model = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Credit_Card'
                      , 'Interest_Rate', 'Credit_Utilization_Ratio', 'Credit_Mix_Bad'
                      , 'Credit_Mix_Good', 'Credit_Mix_Standard', 'Occupation_Accountant', 'Occupation_Architect'
                      , 'Occupation_Developer', 'Occupation_Doctor', 'Occupation_Engineer', 'Occupation_Entrepreneur'
                      , 'Occupation_Journalist', 'Occupation_Lawyer', 'Occupation_Manager', 'Occupation_Mechanic'
                      ,'Occupation_Media_Manager' , 'Occupation_Musician', 'Occupation_Scientist', 'Occupation_Teacher'
                      , 'Occupation_Writer'
                      ] 

target_features = ['Credit_Score_Good', 'Credit_Score_Poor', 'Credit_Score_Standard']

# Getting the size of input size
print(len(features_for_model))


# In[139]:


# Defining data sets
X = encoded_features.toarray()
y = encoded_target.toarray()
# y = to_categorical(df[target_features])
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
model.add(Dense(24, input_dim = X_train.shape[1], activation = 'relu'))
print(X_train.shape[1])
# Adding hidden layer 
model.add(keras.layers.Dense(48, activation="relu"))
model.add(keras.layers.Dense(96, activation="relu"))
model.add(keras.layers.Dense(96, activation="relu"))
model.add(keras.layers.Dense(48, activation="relu"))

# output layer
# For classification tasks, we generally tend to add an activation function in the output ("sigmoid" for binary, and "softmax" for multi-class, etc.).
model.add(keras.layers.Dense(3, activation="softmax"))

print(model.summary())


# In[143]:


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


# In[144]:


# Train the Model
#################
# Training the neural network model requires the following steps:
# Feed the training data to the model. In this example, the training data is in the train_images and train_labels arrays.
# The model learns to associate images and labels.
# You ask the model to make predictions about a test set—in this example, the test_images array.
# Verify that the predictions match the labels from the test_labels array.
# Feed the model
# To start training, call the model.fit method—so called because it "fits" the model to the training data:

model.fit(X_train, y_train, epochs = 12, batch_size = 20)


