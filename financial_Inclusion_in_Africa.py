# import libraries
import pandas as pd
import numpy as np
import streamlit as st
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# load dataset
data = pd.read_csv('./datasets/Financial_inclusion_dataset.csv')

# check the first few rows of the dataset
data.head()

# check the general information about the dataset
info = data.info()

# check the number of rows and columns in the dataset
shape = data.shape
print(f'\n The dataset contains {shape[0]} rows and {shape[1]} columns.')

# view descriptive statistics
desc_stats = data.describe(include='all')
print('\n Descriptive statistics of the dataset:')
print(desc_stats)

# create a pandas profiling report
# profile = ProfileReport(data, title="Expresso churn dataset", explorative=True)
# profile.to_file("Expresso_churn_dataset.html")

# handling missing values
missing_values = data.isnull().sum()
data.dropna(inplace=True)

# check for duplicates
duplicates = data.duplicated().sum()
if duplicates > 0:
    data.drop_duplicates(inplace=True)
    print(f'\n {duplicates} duplicates were removed from the dataset.')
else:
    print('\n No duplicates were found in the dataset.')

# check for outliers using z-score
z_scores = np.abs(stats.zscore(data.select_dtypes(include=[np.number])))
outliers = np.where(z_scores > 3)
if len(outliers[0]) > 0:
    print(f'\n {len(outliers[0])} outliers were found in the dataset.')
    # remove outliers
    data = data[(z_scores < 3).all(axis=1)]
    print(f'\n {len(outliers[0])} outliers were removed from the dataset.')
else:
    print('\n No outliers were found in the dataset.')

# encode categorical features
le =  LabelEncoder()
for col in data.select_dtypes(include=['object']).columns:
    data[col] = le.fit_transform(data[col].astype(str))

# split the dataset into train and test sets
X = data.drop('bank_account', axis=1)
y = data['bank_account']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train and test a machine learning classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('\n Classification report:')
print(classification_report(y_test, y_pred))
print('\n Confusion matrix:')
print(confusion_matrix(y_test, y_pred))

# create a streamlit application
st.title('Expresso Churn Prediction')
st.write('Enter the features values to predict churn:')
st.write(f'The accuracy of the model is {accuracy_score(y_test, y_pred) * 100:.2f}%. ')

# create input fields for the features
features = {}
for col in X.columns:
    if data[col].dtype == 'object':
        features[col] = st.selectbox(col, options=data[col].unique())
    else:
        features[col] = st.number_input(col, value=0)

# create a validation button
if st.button('Validate'):

    # check if all input fields are filled
    if all(value != '' for value in features.values()):        
        input_data = pd.DataFrame(features, index=[0]) # convert the input values to a dataframe
        
        prediction = model.predict(input_data) # make predictions
        if prediction[0] == 1:
            st.write('The customer is likely to have a bank.')
        else:
            st.write('The customer is not likely to have a bank.')
    else:
        st.write('Please fill all the input fields.')