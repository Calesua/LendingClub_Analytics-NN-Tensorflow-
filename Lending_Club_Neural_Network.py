# -*- coding: utf-8 -*-

# -- Sheet --

# # Lending Club
# 
# ## The Data
# 
# We will be using a subset of the LendingClub DataSet obtained from Kaggle: https://www.kaggle.com/wordsforthewise/lending-club
# 
# 
# LendingClub is a US peer-to-peer lending company, headquartered in San Francisco, California.[3] It was the first peer-to-peer lender to register its offerings as securities with the Securities and Exchange Commission (SEC), and to offer loan trading on a secondary market. LendingClub is the world's largest peer-to-peer lending platform.
# 
# ### Our Goal
# 
# Given historical data on loans given out with information on whether or not the borrower defaulted (charge-off), we build a Neural Network model that can predict wether or not a borrower will pay back their loan.
# 
# The "loan_status" column contains our label.


# #### Let's import the packages we will need


import numpy as np
import pandas as pd                          # Allows working with dataframes

import matplotlib.pyplot as plt              # Graphics package
import seaborn as sns                        # Enhanced graphics package
sns.set(style='darkgrid')

import re                                    #regex(regular expression) module

# ML Data preparation

from sklearn.preprocessing import MinMaxScaler                      # Data normalization
from sklearn.model_selection import train_test_split                # Model training/testing data
from sklearn.metrics import classification_report,confusion_matrix  # Model performance metrics
from sklearn.impute import KNNImputer                               # Filling missing data using KNN

#Neural Network

import tensorflow as tf

# Neural network settings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

# Preventing overfitting
from tensorflow.keras.callbacks import EarlyStopping         # Training stopping when there's not enough performance reducing error
from tensorflow.keras.layers import Dropout                  # Rnadomly drops nodes connections to prevent overfitting

# ## 1. Data Overview
# #### Let's load the data and try to visualize some general information in the dataset


# #### 'lending_club_info.csv' contains a description of the variables whose values are given in the historic file 'lending_club_loan_two.csv'


lc_info = pd.read_csv('../DATA/lending_club_info.csv')
pd.options.display.max_colwidth = 150 # This sets the amount length of string to be shown to 150 chars
lc_info

lc_loan = pd.read_csv('../DATA/lending_club_loan_two.csv')
lc_loan.head()

# Proportion of loans that are defaulted
plt.figure(figsize=(8,2))
sns.countplot(data=lc_loan, x='loan_status')

# Distribution of the amount borrowed for historic loans
plt.figure(figsize=(10,3))
sns.histplot(lc_loan['loan_amnt'], kde=False, bins=25)

# #### As we can see, the historical amount of money borrowed concentrates around 10K USD.


# #### Let's now turn loan_status to a dummy variable so we can check the correlations with each other variable


# We associate each element of loan_status with a 0 or a 1 in the a column
lc_loan['status_dummy'] = (lc_loan['loan_status'].map({'Fully Paid':1,'Charged Off':0})).astype(np.uint8)
lc_loan[['loan_status','status_dummy']].tail(10)

# Plot the correlations
plt.figure(figsize=(10,5))
sns.heatmap(lc_loan.corr(), annot=True)

# ####  Let's see description of the variables that show the greatest correlations


lc_info.set_index('LoanStatNew').loc[['installment', 'pub_rec', 'pub_rec_bankruptcies', 'open_acc', 'total_acc']]

# #### As expected, installment is pretty correlated to the total amount of money borrowed
# #### Again, pub_rec and pub_rc_bankruptcies show similar information about public records about payment default
# #### The relation between open_acc and total_acc is probably not that evident, as having a bigger or smaller record does not imply having more or less credit accounts currently open. This said, I would suggest the following reasons that make this happen (We would need more information to accept or reject them):
# * The more experience a person have with credit lines, the more comfortable it feels to have them and perhaps the more that person can feel in need to have extra money for his projects
# * The older the person gets, the more accounts he would have oppened and because of his age, the more money income he could have managed to get and also (related or not to the latter) the less he would worry about having a money debt


# #### Correlation with status_dummy


# Correlations with the target feature
lc_loan.corr()['status_dummy'].abs().sort_values()

# Distribution with respect to the interest rate and clusterized by payment status
plt.figure(figsize=(8,3))
sns.histplot(lc_loan, x='int_rate',hue='loan_status', multiple='dodge', bins=30, kde=True, kde_kws={'bw_adjust':4}) #kde_kws is a smoothness factor

# #### There's better payment rate for bigger interest rate, although the difference is not significant


# Proportion of Fully paid/Charged off loans for each kind of application type
g = sns.FacetGrid(lc_loan,col='application_type', sharey=False, height=3)
#If we let share y, the 2nd and 3rd graphs would be tiny if compared to the 1st
g.map(sns.histplot,'loan_status')

# #### The direct payment type is the most risky and the joint type is the least


# Dependency of loan_status with the duration of the loan (term column)
sns.displot(lc_loan, x='term', hue='loan_status', height=3.5, aspect=1.2)

# #### The proportion Fully paid vs. Charged off is much greater in loans with a 36 months duration


# Distribution of Fully loan amounts for fully paid and charged off loans
sns.boxplot(data=lc_loan, x='loan_status',y='loan_amnt', width=0.5)

# Distribution of loans respect to the quality score assigned (sub_grade)
plt.figure(figsize=(10,3))
sns.countplot(data=lc_loan, x='sub_grade', hue='loan_status', order=sorted(lc_loan['sub_grade'].unique()))

# #### As shown, the proportion of 'fully paid' vs. 'charged off' loans grows between the A1 and the B3 subgrades, then it keeps reducing at a decreasing rate up to the F3 subgrade and from there it's fairly constant


lc_info[lc_info['LoanStatNew'] == 'initial_list_status'].Description

# #### W Loans are a random set of loans which were initially available for whole purchase for investors


# Dependency of loan_status with the initial list status
plt.figure(figsize=(4,3.5))
sns.histplot(lc_loan, x='initial_list_status', hue='loan_status', multiple='fill')

# ## 2. Data transformation
# ### 2.1. Direct transformations


lc_loan.head()

# #### We need to turn some of the features to numerical type in order to use them in our neural network


# ### emp_length


lc_loan['emp_length'].unique() #There are NaN values

def str_to_num(str):
    '''
    This function returns the input variable if it's a float, 0 if the that variable starts with < and the float number contained
    in the string variable in any other case
    '''
    if type(str)==float:
        return str
    elif str.split()[0]=='<':
        return 0
    else:
        return int(re.findall(r'-?\d+\.?\d*', str)[0])

lc_loan['emp_length'] = lc_loan['emp_length'].apply(str_to_num)
lc_loan['emp_length'].unique()

# ### Same process with 'term'


lc_loan['term'] = lc_loan['term'].apply(str_to_num)
lc_loan['term'].unique()

# ### Subgrade column to numerical 
# #### following the next pattern:
# A1 -> 1.00   B1 -> 2.00   ...   G1 -> 7.00
# <br>A2 -> 1.20   B2 -> 2.20   ...   G2 -> 7.20
# <br>A3 -> 1.40   B3 -> 2.40   ...   G3 -> 7.40
# <br>A4 -> 1.60   B4 -> 2.60   ...   G4 -> 7.60
# <br>A5 -> 1.80   B5 -> 2.80   ...   G5 -> 7.80


lc_loan['sub_grade'].unique()

def alphnum_to_num(grad):
    '''
    This function makes the letter in subgrade correspond to the integer part of an output number and the number in subgrade 
    to the decimal part of the output, i.e., it turns the alphanumerical input to numerical type
    '''
    grad_n=[]
    
    #First, we check the letter
    sep = [char for char in grad]
    if sep[0]=='A':
        grad_n.append(1)
    elif sep[0]=='B':
        grad_n.append(2)
    elif sep[0]=='C':
        grad_n.append(3)
    elif sep[0]=='D':
        grad_n.append(4)
    elif sep[0]=='E':
        grad_n.append(5)
    elif sep[0]=='F':
        grad_n.append(6)
    elif sep[0]=='G':
        grad_n.append(7)
        
    #Now let's check subgrades:
    if sep[1]=='1':
        grad_n.append(0.00)
    elif sep[1]=='2':
        grad_n.append(0.20)
    elif sep[1]=='3':
        grad_n.append(0.40)
    elif sep[1]=='4':
        grad_n.append(0.60)
    elif sep[1]=='5':
        grad_n.append(0.80)
        
    return sum(grad_n)

lc_loan['sub_grade'] = lc_loan['sub_grade'].apply(alphnum_to_num)
lc_loan.sub_grade.isnull().sum() # No null values

# Drop grade column
lc_loan.drop(['grade'], axis=1, inplace=True)

lc_loan.head()

# ## 2.2. Create dummy variables for categorical features
# ### application_type


lc_loan['application_type'].unique()

lc_loan = pd.get_dummies(data=lc_loan, prefix='dum', columns=['application_type'], drop_first=True)
lc_loan.head()

lc_loan.columns

# We can rename the new columns if we want
lc_loan.rename(
    columns={"dum_INDIVIDUAL": "Individual", "dum_JOINT": "Joint"},
    inplace=True
)

# ### emp_title


# Number of unique employments
lc_loan['emp_title'].nunique()

lc_loan['emp_title'].value_counts()[:30]

# #### As there are too many different jobs, we drop emp_title


lc_loan.drop('emp_title', axis=1, inplace=True)

# ### title and purpose columns


lc_loan['title'].head(10)

lc_loan['purpose'].head(10)

# #### title and purpose show about the same information. We can drop title


lc_loan.drop('title', axis=1, inplace=True)

# Number of unique values
lc_loan['purpose'].nunique()

# #### Let's make it a dummy variable


lc_loan = pd.get_dummies(data=lc_loan, prefix='dum', columns=['purpose'], drop_first=True)

lc_loan.info()

# ### home_ownership


lc_loan['home_ownership'].value_counts()

# ##### Let's join Other, None and Any into 1 category called Other


lc_loan['home_own'] = lc_loan['home_ownership']

lc_loan.head(30)

lc_loan['home_own'] = lc_loan['home_ownership'].apply(lambda x: 'OTHER' if x in ['NONE', 'ANY'] else x)
# The lambda function used returns OTHER if the input is NONE or ANY

lc_loan['home_own'].value_counts()

lc_loan = pd.get_dummies(data=lc_loan, prefix='dum', columns=['home_own'], drop_first=True)
lc_loan.drop(['home_ownership'], axis=1, inplace=True)          #Drop original column

lc_loan.info()

# ### verification_status


lc_loan['verification_status'].value_counts()

# Make it a dummy
lc_loan = pd.get_dummies(data=lc_loan, prefix='dum', columns=['verification_status'], drop_first=True)

# ### Issue_d


# Let's recall what issue_d means


lc_info.columns

lc_info[lc_info['LoanStatNew'] == 'issue_d'].Description

lc_loan['issue_d']

# #### If we can broke down issue_d into Year and Month, would both variables be treated lenearly?


# Save the loan month
lc_loan['loan_month'] = lc_loan['issue_d'].apply(lambda x: x[-8:-5]) # 8th position to 5th position from the end

lc_loan['loan_month'].value_counts()

# Save the loan year
lc_loan['loan_year'] = lc_loan['issue_d'].apply(lambda x: x[-4:]).astype(np.uint16)

lc_loan['loan_year'].value_counts()

# ##### We have the 12 months of a Year and 10 diferent years. We can treat the "month variable categorically" as the month in which future loans will be requested will fall into one of the existing categories (the 12 months), but regarding the years, new loans will occur in future years and the values of the past won't be repeated (maybe the last year will if it's not over yet). Because of that I will try "years linearly"


# #### Turn month into dummy


# Turn month into dummy
lc_loan = pd.get_dummies(data=lc_loan, prefix='dum_loan', columns=['loan_month'], drop_first=True)
lc_loan.drop(['issue_d'], axis=1, inplace=True)        # Drop original column

# Let's have a look
lc_loan.info()

# ### earliest_cr_line


lc_loan['earliest_cr_line']

# Description
lc_info[lc_info['LoanStatNew'] == 'earliest_cr_line']

# #### Let's do the same we did with issue_d


# Save the first credit line year
lc_loan['earliest_cr_year'] = lc_loan['earliest_cr_line'].apply(lambda x: x[-4:]).astype(np.uint16)

# Save the first credit line month
lc_loan['earliest_cr_month'] = lc_loan['earliest_cr_line'].apply(lambda x: x[-8:-5])

# Turn month into dummy
lc_loan = pd.get_dummies(data=lc_loan, prefix='dum_ear_cr', columns=['earliest_cr_month'], drop_first=True)
lc_loan.drop(['earliest_cr_line'], axis=1, inplace=True)        # Drop original column

# ### Initial list status


# Turn month into dummy
lc_loan = pd.get_dummies(data=lc_loan, prefix='dum', columns=['initial_list_status'], drop_first=True)

# ### Address


lc_loan['address'].head()

lc_loan['zip'] = lc_loan['address'].apply(lambda x: x[-5:])

lc_loan['address'].apply(lambda x: x[-8:])

# How many State-zip code combinations are there?
lc_loan['address'].apply(lambda x: x[-8:]).nunique()

# How many different zip codes?
lc_loan['zip'].nunique()

# Let's check how many States there are in the data


lc_loan['address'].apply(lambda x: x[-8:-6])

# How many different States?
lc_loan['address'].apply(lambda x: x[-8:-6]).nunique()

# #### I will just use the zipcode for now because there are too many states


lc_loan = pd.get_dummies(data=lc_loan, prefix='dum_zip', columns=['zip'], drop_first=True)
lc_loan.drop('address', axis=1, inplace=True)          # Drop the original column
lc_loan.head()

lc_loan.info()

# ## 3. Missing Data


# Number of missing values (NaNs)
lc_loan.isna().sum().sort_values()

# Missing values ratio
lc_loan.isna().sum().sort_values()/len(lc_loan)*100

# #### As the number of NaNs in "revol_util" and "pub_rec_bankruptcies" is relatively low, we can just drop the rows that contain those NaNs and fill the missing data in the other two columns


lc_loan.drop(lc_loan[lc_loan['revol_util'].isna()].index, inplace=True)
lc_loan.drop(lc_loan[lc_loan['pub_rec_bankruptcies'].isna()].index, inplace=True)
lc_loan.isna().sum().sort_values()/len(lc_loan)*100

# # 4. Scaling the data


lc_loan.drop(('loan_status'), axis=1, inplace=True)
lc_loan.info()

# Create a dataframe without the dummy columns to scale just those variables

# use numpy r_ to concatenate slices
dumms = lc_loan.iloc[:,np.r_[16:36, 37:48, 49:70, 15]]
noDum = lc_loan.iloc[:,np.r_[:15, 36, 48]]

# We should scale the data as KNN is a distance based algorithm and it reduces biases

# Instanciate the scaler model and scale the data
scaler = MinMaxScaler()
scaled_noDum = pd.DataFrame(scaler.fit_transform(noDum), columns = noDum.columns)

scaled_noDum.describe()

# #### It looks good. 


# Add the dummy columns
scaled_df = scaled_noDum.join(dumms, on=dumms.index)
scaled_df.info()

# ## 5. Neural Network


early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

scaled_df.shape # Width of 48 neurons, as the last one is the target

# Setting up the Neural Network

model = Sequential()

model.add(Dense(units=70, activation='relu'))   # Input layer
model.add(Dropout(0.2))
model.add(Dense(units=35, activation='relu'))   # Hidden Layer 1
model.add(Dropout(0.2))
model.add(Dense(units=12, activation='relu'))   # Hidden Layer 2
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid')) # Output layer

# For a binary classification problem : binary_crossentropy
model.compile(loss='binary_crossentropy', optimizer='adam')

# ### 5.1. Model training/testing


# #### After trying many runs with different NN model hyperparameters and number of neighbors for the imputation ( in which the results didn't change too much), the best result was achieved with the following setup


imputer = KNNImputer(n_neighbors=2)
imputed_df = imputer.fit_transform(scaled_df)  # This Outputs an array

# Neural networks work with arrays
y = imputed_df[:,69]                           # status_dummy
X = imputed_df[:,:69]                          # rest of columns
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)

model.fit(x=X_train, 
          y=y_train, 
          epochs=150,                  # number times that the learning algorithm will work through the entire training dataset
          batch_size=256,              # number of samples processed before the model is updated
          validation_data=(X_test, y_test), verbose=1,
          callbacks=[early_stop]
          )

# ### Model performance


# Create global variables containing the predictions
pred2 = (model.predict(X_test) > 0.5).astype("int32")

print(classification_report(y_test.astype(int),pred2))

# #### It would be better that the recall for "0" (charged-off loans) was higher, as this is the model capability to detect all the loans of this kind. Despite that, this is a normal result, given the big quantity difference between fully-paid and charged-off loans present in the data.


