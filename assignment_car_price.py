


# importing librarys
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#reading the car dataset
df = pd.read_csv('CAR DETAILS.csv')

#checking top 5 rows to understand the dataset
df.head()

#checking if there is any null values
df.isnull().sum()

#checking if there are duplicate rows in the dataset
df.duplicated().sum()

#dropping duplicate values
df.drop_duplicates(inplace= True)

df.duplicated().sum()

#displaying all the column names
df.columns

#checking datatypes
df.dtypes

#checking how many rows and columns are there in the dataset, there are 3577 rows and 8 columns.
print(df.shape)

#printing value counts of the name columns
print(df['name'].value_counts())
print(df['name'].nunique())

#instead of using name with model we will use just company name so i am creating a new column named company
df['company']= df['name'].str.split().str[0]

df.head()

#creating a copy of the dataset
df1= df.copy()

#checking value count
print(df['company'].value_counts())
print(df['company'].nunique())

#droping those names which have value count of less than 5 because sample size is very low.
value_count= df['company'].value_counts()
value_count= value_count[value_count<=5]
value_count

#droping those names which have value count of less than 5 because sample size is very low.
rows_to_drop= list(value_count.index)
rows_to_drop

#droping those names which have value count of less than 5 because sample size is very low.
df.drop(df[df['company'].isin(rows_to_drop)].index, axis=0,inplace= True)

df.head()

#creating a new column car age by using year column
df['car_age']= 2023-df['year']

#dropping name column
df.drop('name',axis=1, inplace= True)

df.head()



# creating a for loop to check unique values of all the columns if values of the unique value is less than 10 we will print value count of
#unique values too else we will just show how many unique values are there.
for i in df.columns:
    # Check if the number of unique values in the column is less than 10
    if df[i].nunique() < 10:
        # Print column information, including the column name, data type, number of unique values, and value counts
        print(f'The column "{i}" is __{df[i].dtype}__ \nhas __{df[i].nunique()}__ unique values: \n{df[i].value_counts()}')
        # Print a separator line
        print(20*'==')
    else:
        # Print column information, including the column name, data type, and number of unique values
        print(f'The column "{i}" is __{df[i].dtype}__ \nhas __{df[i].nunique()}__ unique values')
        # Print a separator line
        print(20*'==')

#creating a new variable with column company','owner','transmission','seller_type','fuel','year'
df_c= df[['company','owner','transmission','seller_type','fuel','year']]

#creating a for loop to generate count plot charts for the above mentioned columns
x = 0
fig = plt.figure(figsize=(20, 25))  # Create a figure with the specified size
plt.subplots_adjust(wspace=0.4)  # Adjust the spacing between subplots

# Iterate over each column in the DataFrame df_c
for i in df_c.columns:
    ax = plt.subplot(321 + x)  # Create a subplot at position 321+x in a 3x2 grid
    ax = sns.countplot(data=df_c, y=i)  # Create a countplot using data from df_c with y-axis representing the column values
    plt.grid(axis='x')  # Add gridlines along the x-axis
    ax.set_title(f'Distribution of {i}')  # Set the title of the subplot
    x += 1  # Increment x to move to the next subplot position

"""
*   most of the cars are manufatured in years between 2008 -2019.
*   most of the cars uses diesel or petrol, very few car uses cng and lpg.
*   most sold brand is maruti, hyundai is on second number."""

#company vs selling price bar plot
plt.figure(figsize=(20,10))
sns.barplot(data= df, x='company', y= 'selling_price')

"""

1.   mercedes-benz, audi,bmw got the maximum selling price
2.   tata, chvrolet, are getting sold for lowest price according to this dataset.tata and mahindra got sold almost same times still mahindra's car's resale value is higher.

"""

plt.figure(figsize=(20,10))
sns.barplot(data= df, x='company', y= 'selling_price', hue= 'owner')

"""

1.   according to the data mercedes-benz is getting sold for just two times,and audi and bmw are also barely getting sold for the third time.
2.  toyota is getting sold for a very good price when compared with other same categories brands, suggesting that resale value of toyota is higher than others.
3.   For test drive related cars,people bought honda,ford, renault and volkwagen
only, there must be a reason behind that too


"""

#bar chart using company name,and selling price grouped by fuel type
plt.figure(figsize=(20,10))
sns.barplot(data= df, x='company', y= 'selling_price', hue= 'fuel')

"""cars with diesel option got higher resale value, it is high because of the two 
reason- first is diesel varient cost more and second is there is high demand of diesel cars because diesel cost less than the petrol.


*   only maruti, hyundai, tata and chevrolet's lpg and cng cars got sold which means they were the one people bought, which tells that only these company focused on the different types of cars.



"""

#price vs fuel type chart
plt.figure(figsize=(20,6))
sns.barplot(data= df, x='fuel', y = 'selling_price')

#seller type vs selling price
plt.figure(figsize=(15,8))
sns.barplot(data= df, x='seller_type',y='selling_price')

"""trustmark dealer sold cars for higher price but why we will analyse that in next few charts"""

plt.figure(figsize=(20,10))
sns.barplot(data= df, x='seller_type',y='selling_price',hue='owner')

"""reson 1- while dealer and individual seller sold all types of cars, trustmark dealer sold only first hand and second hand cars."""

#checking count of how many cars each of these sellers sold
plt.figure(figsize=(20,10))
sns.countplot(data= df, x='company',hue ='seller_type')

"""reason 2- trustmark dealers sold very few cars"""

#count of cars and its company sold by all the three types of dealers
a= df.groupby(['seller_type'])['company'].value_counts()
a

plt.figure(figsize=(15,5))
sns.barplot(data= df, x= 'seller_type', y='km_driven')

"""reson 3- trustmark dealer sold only thoose cars which were not driven that much,
individual sold almost all types of cars
, dealer sold cars below 60000km
and trustmark dealer sold those car which were below the mark of 45000km or below.
"""

#year wise seller type
plt.figure(figsize=(15,8))
sns.countplot(data= df, x= 'year',hue='seller_type')

"""reason 4- trustmark dealers are only selling car which got manufatured after 2013 which is another reason why there got sold cars for higher price."""

plt.figure(figsize=(15,8))
sns.countplot(data= df, x= 'car_age',hue='seller_type')

#plotig car selling price based on  its transmission
plt.figure(figsize=(15,6))
sns.barplot(data= df, x= 'transmission', y='selling_price', hue= 'owner')

"""automatic cars were sold for way more than manual cars, the main reason behind that is high cost of automatic cars.

but why test drive cars are getting sold for that much?
"""

#ploting transmission and km driven grouped by owner type
plt.figure(figsize=(15,5))
sns.barplot(data= df, x= 'transmission', y='km_driven',hue= 'owner')
plt.grid()
plt.show()

"""average KM at manual cars are getting sold for the first time is above 60000
while in automatic cars it is around 52000 there can be multiple reason for that reasons can be- Automatic cars owners are facing more problems or it can be just because people with good financial condition usually buys these type of cars and maybe they are selling them because they got bored.

look at that test drive cars, why they are getting sold for that much?because of the km driven, they barely crossed 15000 km mark, almost all of the cars are below 15000 marks
"""

# Print the maximum value of the 'km_driven' column
print(df['km_driven'].max())

# Print the minimum value of the 'km_driven' column
print(df['km_driven'].min())

# Calculate 0.95, 0.99 percentiles for the 'km_driven' column
df['km_driven'].describe(percentiles=[0.95, 0.99])

def km_range(km):
    
    #writing a func that Determines the range category for a given kilometer value.   
    if km < 20000:
        return '0-19999'
    elif km < 40000:
        return '20000-39999'
    elif km < 60000:
        return '40000-59999'
    elif km < 80000:
        return '60000-79999'
    elif km < 100000:
        return '80000-99999'
    elif km < 120000:
        return '100000-119999'
    elif km < 140000:
        return '120000-139999'
    elif km < 160000:
        return '140000-159999'
    elif km < 180000:
        return '160000-179999'
    else:
        return '200000+'

# Add a new column 'km_range' to the DataFrame based on the 'km_driven' values
df['km_range'] = df['km_driven'].apply(km_range)
# The apply() function is used to apply the km_range() function to each element in the 'km_driven' column and assign the returned value to the 'km_range' column

# Set the figure size for the plot
plt.figure(figsize=(15, 6))

# Create a bar plot using seaborn
sns.barplot(data=df, x='km_range', y='selling_price')
# This code generates a bar plot to visualize the relationship between the 'km_range' and 'selling_price' columns in the DataFrame 'df'.

# This code generates a bar plot to visualize the relationship between the 'km_driven' and 'company' columns in the DataFrame 'df'.
plt.figure(figsize=(15,6))
sns.barplot(data= df, x='company', y= 'km_driven' )

"""toyota is the most driven car maybe because of its good quality work

"""

plt.figure(figsize=(15, 6))
sns.barplot(data=df, x='km_range', y='selling_price', hue='fuel')
# This code generates a bar plot to visualize the relationship between the 'km_range', 'selling_price', and 'fuel' columns in the DataFrame 'df'.

"""even after being in the same km range diesel cars are getting sold for more money when compared to another cars

"""

plt.figure(figsize=(15, 6))
sns.barplot(data=df, x='fuel', y='selling_price', hue='km_range')
# This code generates a bar plot to visualize the relationship between the 'fuel', 'selling_price', and 'km_range' columns in the DataFrame 'df'.

# Count the occurrences of each fuel type in the 'fuel' column
df['fuel'].value_counts()

# Drop rows where the fuel type is 'Electric'
df.drop(df[df['fuel'] == 'Electric'].index, axis=0, inplace=True)

df['fuel'].value_counts()

# Calculate the correlation matrix
corr = df.corr()
sns.heatmap(corr, annot=True)
# This code generates a heatmap using the correlation matrix 'corr' to visualize the relationships between variables in the DataFrame 'df'.
plt.show()

# Set the figure size for the plot
plt.figure(figsize=(15, 6))

# Create a distribution plot using seaborn
sns.displot(df['km_driven'])
# This code generates a distribution plot to visualize the distribution of values in the 'km_driven' column of the DataFrame 'df'.
# Rotate the x-axis labels by 90 degrees
plt.xticks(rotation=90)
# This code rotates the x-axis labels by 90 degrees to improve readability when there are many values.
# Display the plot
plt.show()

sns.displot(df['selling_price'])
# This code generates a distribution plot to visualize the distribution of values in the 'selling_price' column of the DataFrame 'df'.
# The plot shows the frequency of different selling price values.
plt.show()

df.describe(percentiles=[0.005,0.01,0.02,0.03,0.97,0.98,0.99,0.995]).T

"""#**`checking and fixing outliers.`**"""

#creating a box plot of the year
sns.boxplot(x=df['year'])

print(df[df['year']<2000].shape)

sns.boxplot(x=df['selling_price'])

# Print the shape of the DataFrame where 'selling_price' values are less than 45000
print(df[df['selling_price'] < 45000].shape)
# Print the shape of the DataFrame where 'selling_price' values are greater than 3100000
print(df[df['selling_price'] > 3100000].shape)

sns.boxplot(x=df['km_driven'])

# Print the shape of the DataFrame where 'km_driven' values are less than 1010
print(df[df['km_driven'] < 1010].shape)

# Print the shape of the DataFrame where 'km_driven' values are greater than 257420.0
print(df[df['km_driven'] > 257420.0].shape)

df.shape

# Set a lower limit for 'km_driven' column values
df['km_driven'] = np.where(df['km_driven'] < 1010, 1010, df['km_driven'])
# This code sets a lower limit of 1010 for the values in the 'km_driven' column.
# If a value is less than 1010, it is replaced with 1010.

# Set an upper limit for 'km_driven' column values
df['km_driven'] = np.where(df['km_driven'] > 252550, 252550, df['km_driven'])
# This code sets an upper limit of 252550 for the values in the 'km_driven' column.
# If a value is greater than 252550, it is replaced with 252550.

# Set a lower limit for 'year' column values
df['year'] = np.where(df['year'] < 1999, 1999, df['year'])
# This code sets a lower limit of 1999 for the values in the 'year' column.
# If a value is less than 1999, it is replaced with 1999.

# Set an upper limit for 'selling_price' column values
df['selling_price'] = np.where(df['selling_price'] > 3100000, 3100000., df['selling_price'])
# This code sets an upper limit of 3100000 for the values in the 'selling_price' column.
# If a value is greater than 3100000, it is replaced with 3100000.

# Set a lower limit for 'selling_price' column values
df['selling_price'] = np.where(df['selling_price'] < 45000, 45000, df['selling_price'])
# This code sets a lower limit of 45000 for the values in the 'selling_price' column.
# If a value is less than 45000, it is replaced with 45000.

from sklearn.model_selection import train_test_split

df.drop('car_age',axis=1, inplace= True)
#droping car age column

df.drop('km_range',axis= 1, inplace = True)
#droping km range column

x= df.drop('selling_price', axis= 1)
y= df['selling_price']
#creating two variables x and y to train and test dataset

xtrain,xtest,ytrain,ytest= train_test_split(x,y, test_size= 0.30, random_state= 42)
#spliting the data into train and test

print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)

from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score

d= {'model':[],'mse':[],'rmse':[],'mae':[],'r2s':[]}
#creating a dict to store the performance data

def model_eval(model_name, ytest, ypred):

    #Evaluate a model's performance using various metrics and store the results in a dictionary.
    mse = mean_squared_error(ytest, ypred)
    mae = mean_absolute_error(ytest, ypred)
    rmse = np.sqrt(mse)
    r2s = r2_score(ytest, ypred)
    
    # Print the evaluation metrics
    print('mse:', mse)
    print('rmse:', rmse)
    print('r2s:', r2s)
    print('mae:', mae)
    
    # Store the results in a dictionary
    d['model'].append(model_name)
    d['mse'].append(mse)
    d['rmse'].append(rmse)
    d['r2s'].append(r2s)
    d['mae'].append(mae)



# Import the necessary modules and classes for regression modeling
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

xtrain.dtypes
#checking data types of xtrain

# Step 1: ColumnTransformer for categorical feature encoding and passthrough
step1 = ColumnTransformer(transformers=[('ohe', OneHotEncoder(drop='first', sparse=False), [2, 3, 4, 5, 6])],
                          remainder='passthrough')
# This step applies OneHotEncoder to the categorical features at indices 2, 3, 4, 5, and 6 in the input data.
# The 'drop' parameter is set to 'first' to drop the first category of each encoded feature.
# The 'sparse' parameter is set to False to return a dense array.

# Step 2: Linear Regression model
step2 = LinearRegression()
# This step creates an instance of the LinearRegression model.

# Create a pipeline with the defined steps
pipelr = Pipeline([('step1', step1), ('step2', step2)])
# This pipeline combines the ColumnTransformer and LinearRegression into a single object.

# Fit the pipeline to the training data
pipelr.fit(xtrain, ytrain)
# This step fits the pipeline to the training data, applying the transformations and training the linear regression model.

# Predict the target variable using the pipeline
ypredlr = pipelr.predict(xtest)
# This step uses the fitted pipeline to predict the target variable using the test data.

# Evaluate the performance of the linear regression model
model_eval('linear_reg', ytest, ypredlr)
# This step calls the 'model_eval' function to evaluate the performance of the linear regression model.

# Step 1: ColumnTransformer for categorical feature encoding and passthrough
step1 = ColumnTransformer(transformers=[('ohe', OneHotEncoder(drop='first', handle_unknown='ignore', sparse=False), [2, 3, 4, 5, 6])],
                          remainder='passthrough')
# This step applies OneHotEncoder to the categorical features at indices 2, 3, 4, 5, and 6 in the input data.
# The 'drop' parameter is set to 'first' to drop the first category of each encoded feature.
# The 'handle_unknown' parameter is set to 'ignore' to handle unknown categories during encoding.
# The 'sparse' parameter is set to False to return a dense array.

# Step 2: Ridge Regression model with alpha=2.1
step2 = Ridge(alpha=2.1)
# This step creates an instance of the Ridge Regression model with the specified alpha value.

# Create a pipeline with the defined steps
piperid = Pipeline([('step1', step1), ('step2', step2)])
# This pipeline combines the ColumnTransformer and Ridge Regression into a single object.

# Fit the pipeline to the training data
piperid.fit(xtrain, ytrain)
# This step fits the pipeline to the training data, applying the transformations and training the Ridge Regression model.

# Predict the target variable using the pipeline
ypredrid = piperid.predict(xtest)
# This step uses the fitted pipeline to predict the target variable using the test data.

# Evaluate the performance of the Ridge Regression model
model_eval('Ridge', ytest, ypredrid)
# This step calls the 'model_eval' function to evaluate the performance of the Ridge Regression model.

# Step 1: ColumnTransformer for categorical feature encoding and passthrough
step1 = ColumnTransformer(transformers=[('ohe', OneHotEncoder(drop='first', handle_unknown='ignore', sparse=False), [2, 3, 4, 5, 6])],
                          remainder='passthrough')
# This step applies OneHotEncoder to the categorical features at indices 2, 3, 4, 5, and 6 in the input data.
# The 'drop' parameter is set to 'first' to drop the first category of each encoded feature.
# The 'handle_unknown' parameter is set to 'ignore' to handle unknown categories during encoding.
# The 'sparse' parameter is set to False to return a dense array.

# Step 2: Ridge Regression model with alpha=2.1
step2 = Ridge(alpha=2.1)
# This step creates an instance of the Ridge Regression model with the specified alpha value.

# Create a pipeline with the defined steps
piperid = Pipeline([('step1', step1), ('step2', step2)])
# This pipeline combines the ColumnTransformer and Ridge Regression into a single object.

# Fit the pipeline to the training data
piperid.fit(xtrain, ytrain)
# This step fits the pipeline to the training data, applying the transformations and training the Ridge Regression model.

# Predict the target variable using the pipeline
ypredrid = piperid.predict(xtest)
# This step uses the fitted pipeline to predict the target variable using the test data.

# Evaluate the performance of the Ridge Regression model
model_eval('Ridge', ytest, ypredrid)
# This step calls the 'model_eval' function to evaluate the performance of the Ridge Regression model.

# Step 1: ColumnTransformer for categorical feature encoding and passthrough
step1 = ColumnTransformer(transformers=[('ohe', OneHotEncoder(drop='first', handle_unknown='ignore', sparse=False), [2, 3, 4, 5, 6])],
                          remainder='passthrough')
# This step applies OneHotEncoder to the categorical features at indices 2, 3, 4, 5, and 6 in the input data.
# The 'drop' parameter is set to 'first' to drop the first category of each encoded feature.
# The 'handle_unknown' parameter is set to 'ignore' to handle unknown categories during encoding.
# The 'sparse' parameter is set to False to return a dense array.

# Step 2: Decision Tree Regression model with specified parameters
step2 = DecisionTreeRegressor(max_depth=15, min_samples_split=11, random_state=5)
# This step creates an instance of the Decision Tree Regression model with the specified parameters.
# The 'max_depth' parameter sets the maximum depth of the decision tree.
# The 'min_samples_split' parameter sets the minimum number of samples required to split an internal node.
# The 'random_state' parameter sets the random seed for reproducibility.

# Create a pipeline with the defined steps
pipedt = Pipeline([('step1', step1), ('step2', step2)])
# This pipeline combines the ColumnTransformer and Decision Tree Regression into a single object.

# Fit the pipeline to the training data
pipedt.fit(xtrain, ytrain)
# This step fits the pipeline to the training data, applying the transformations and training the Decision Tree Regression model.

# Predict the target variable using the pipeline
ypreddt = pipedt.predict(xtest)
# This step uses the fitted pipeline to predict the target variable using the test data.

# Evaluate the performance of the Decision Tree Regression model
model_eval('dt', ytest, ypreddt)
# This step calls the 'model_eval' function to evaluate the performance of the Decision Tree Regression model.
# It compares the predicted values (ypreddt) with the true values (ytest) of the target variable.
# The evaluation metrics are calculated and stored in the evaluation dictionary.

from sklearn.ensemble import BaggingRegressor

# Define the preprocessing steps and the BaggingRegressor with DecisionTreeRegressor base estimator
step1 = ColumnTransformer(transformers=[('ohe', OneHotEncoder(drop='first', handle_unknown='ignore', sparse=False), [2, 3, 4, 5, 6])],
                          remainder='passthrough')
step2 = BaggingRegressor(base_estimator=DecisionTreeRegressor(max_depth=100, min_samples_split=110, random_state=5),
                         n_estimators=15, max_samples=xtrain.shape[0], max_features=xtrain.shape[1], random_state=2022)

# Create the pipeline by combining the preprocessing steps and the model
pipebrdt = Pipeline([('step1', step1), ('step2', step2)])

# Fit the pipeline on the training data
pipebrdt.fit(xtrain, ytrain)

# Make predictions on the test data
ypredbrdt = pipebrdt.predict(xtest)

# Evaluate the model using the model_eval function
model_eval('bgdt', ytest, ypredbrdt)

# Define the preprocessing steps and the RandomForestRegressor model
step1 = ColumnTransformer(transformers=[('ohe', OneHotEncoder(drop='first', sparse=False), [2, 3, 4, 5, 6])],
                          remainder='passthrough')
step2 = RandomForestRegressor(n_estimators=100, max_depth=20, min_samples_split=9, random_state=5)

# Create the pipeline by combining the preprocessing steps and the model
piperf = Pipeline([('step1', step1), ('step2', step2)])

# Fit the pipeline on the training data
piperf.fit(xtrain, ytrain)

# Make predictions on the test data
ypredrf = piperf.predict(xtest)

# Evaluate the model using the model_eval function
model_eval('rf', ytest, ypredrf)

# Retrieve the feature names after one-hot encoding
feature_names = step1.named_transformers_['ohe'].get_feature_names_out()
# This step retrieves the feature names after applying one-hot encoding in the ColumnTransformer.

feature_names

# Step 1: ColumnTransformer for categorical feature encoding and passthrough
step1 = ColumnTransformer(transformers=[('ohe', OneHotEncoder(drop='first', handle_unknown='ignore', sparse=False), [2, 3, 4, 5, 6])],
                          remainder='passthrough')
# This step applies OneHotEncoder to the categorical features at indices 2, 3, 4, 5, and 6 in the input data.

# Step 2: AdaBoost Regression with RandomForestRegressor as the base estimator
step2 = AdaBoostRegressor(base_estimator=RandomForestRegressor(n_estimators=30, max_depth=15, min_samples_split=20, random_state=5),
                          n_estimators=15)
# This step creates an instance of the AdaBoostRegressor with a RandomForestRegressor as the base estimator.
# The base estimator is a RandomForestRegressor with specified parameters.

# Create a pipeline with the defined steps
pipeadar = Pipeline([('step1', step1), ('step2', step2)])
# This pipeline combines the ColumnTransformer and AdaBoost Regression into a single object.

# Fit the pipeline to the training data
pipeadar.fit(xtrain, ytrain)

# Predict the target variable using the pipeline
ypredadar = pipeadar.predict(xtest)

# Evaluate the performance of the AdaBoost Regression with RandomForest model
model_eval('adarf', ytest, ypredadar)

from sklearn.ensemble import GradientBoostingRegressor

# Step 1: ColumnTransformer for categorical feature encoding and passthrough
step1 = ColumnTransformer(transformers=[('ohe', OneHotEncoder(drop='first',handle_unknown='ignore', sparse=False), [2, 3, 4, 5, 6])],
                          remainder='passthrough')
# This step applies OneHotEncoder to the categorical features at indices 2, 3, 4, 5, and 6 in the input data.
# The 'drop' parameter is set to 'first' to drop the first category of each encoded feature.
# The 'sparse' parameter is set to False to return a dense array.

# Step 2: Gradient Boosting Regression model with specified parameters
step2 = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
# This step creates an instance of the Gradient Boosting Regression model with the specified parameters.
# The 'n_estimators' parameter sets the number of boosting stages.
# The 'learning_rate' parameter controls the contribution of each tree.
# The 'max_depth' parameter sets the maximum depth of each tree.
# The 'random_state' parameter sets the random seed for reproducibility.

# Create a pipeline with the defined steps
pipegb = Pipeline([('step1', step1), ('step2', step2)])
# This pipeline combines the ColumnTransformer and Gradient Boosting Regression into a single object.

# Fit the pipeline to the training data
pipegb.fit(xtrain, ytrain)
# This step fits the pipeline to the training data, applying the transformations and training the Gradient Boosting Regression model.

# Predict the target variable using the pipeline
ypredgb = pipegb.predict(xtest)
# This step uses the fitted pipeline to predict the target variable using the test data.

# Evaluate the performance of the Gradient Boosting Regression model
model_eval('gb', ytest, ypredgb)
# This step calls the 'model_eval' function to evaluate the performance of the Gradient Boosting Regression model.
# It compares the predicted values (ypredgb) with the true values (ytest) of the target variable.
# The evaluation metrics are calculated and stored in the evaluation dictionary.

print(d)

model_eval_score= pd.DataFrame(d)
#converting d into a data frame

model_eval_score
#all the scores in one table



import pickle

import pickle

# Save the pipeline object to a file using pickle
pickle.dump(piperf, open('pipe_rf_car.pkl', 'wb'))

import pickle

# Save the DataFrame object to a file using pickle
pickle.dump(df, open('carsellingdata.pkl', 'wb'))
# This step uses the 'pickle.dump()' function to save the 'df' DataFrame object to a file named 'carsellingdata.pkl'.

import pickle

# Load the pipeline object from the file using pickle
load = pickle.load(open('pipe_rf_car.pkl', 'rb'))
# This step uses the 'pickle.load()' function to load the pipeline object from the file 'pipe_rf_car.pkl'.

print(type(load))

# Random sample of 20 rows from the DataFrame
random20 = df.sample(n=20)

prediction= load.predict(random20)
#predicting price for random 20 rows

prediction

prediction.astype(int)
#converting result into int

random20['predicted_price']= prediction.astype(int)
#creating a new column

random20

random20=random20[['selling_price','predicted_price']]
#selecting only two columns from the random 20 data set

# Calculate the residual and percentage difference for the randomly sampled DataFrame
random20['Residual'] = random20['selling_price'] - random20['predicted_price']
random20['Difference%'] = np.absolute(random20['Residual'] / random20['selling_price'] * 100)
# This step calculates the residual by subtracting the predicted price from the selling price for each row in the DataFrame.
# The result is assigned to the 'Residual' column of the DataFrame.
# The percentage difference is calculated by dividing the absolute value of the residual by the selling price and multiplying by 100.
# The result is assigned to the 'Difference%' column of the DataFrame.

# The 'random20' DataFrame now includes the 'Residual' column and the 'Difference%' column, representing the residual and percentage difference for each row.

random20
