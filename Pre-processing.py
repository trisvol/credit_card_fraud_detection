#%% [markdown]
# # Credit Card Fraud Detection  
# 
# ## Group 12

#%%
#Libraries
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec # to do the grid of plots

#%% [markdown]
# ###  Loading Data 

#%%

# reading data from csv file
df = pd.read_csv('creditcard.csv')


#%%
# prinitng 1st 5 rows with headings  
data_top = df.head()
data_top.to_csv("data_head.csv")
df.head(10)
#There are no null values in the dataset.

#%% [markdown]
# ## Salient Features of data:  
# Total 31 attributes (including class)  
# Time is a Discrete-valued numeric attribute.  
# V1 to V28 are Principal Components of the orginial dataset not avaliable to us.  
# They are a result of Principal Component Analysis.   
# They are continuous valued numeric attributes. We cannot say whether they are ratio-scaled or interval-scaled   
# Amount is a continuous-valued numeric attribute.  
# Class is a discrete-valued Binary attribute that takes value 0 for non-fraudulent transaction and 1 for fraud transaction.  
# V1 to V28 are distributed aroud 0 and are scaled.  
# From V1 to V28, the variance of attributes decreases from left to right, as expected from a PCA output.  

#%%
# prinitng 5 number summary, basic info about the data
data_summary = df.describe()
data_summary.to_csv("data_summary.csv")
df.describe()

#%% [markdown]
# ### Checking on Amount and Time Data

#%%

df[['Time', 'Amount']].describe()
# Time and Amount are not scaled.

#%% [markdown]
# # Visualizing Data Distribution

#%%
# Time and Amount Distribution

print('Non Fraudulent: ', round(df['Class'].value_counts()[0]/len(df) * 100,3), '% of the dataset')
print('Fraudulent: ', round(df['Class'].value_counts()[1]/len(df) * 100,3), '% of the dataset')

# colors = ["#0101DF", "#DF0101"]
# sns.countplot('Class', data=df, palette=colors)
# plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)
# plt.show()

fig, ax = plt.subplots(1, 2, figsize=(18,4))

amount_val = df['Amount'].values
time_val = df['Time'].values

sns.distplot(amount_val, ax=ax[0], color='r')
ax[0].set_title('Distribution of Transaction Amount', fontsize=14)
ax[0].set_xlim([min(amount_val), max(amount_val)])

sns.distplot(time_val, ax=ax[1], color='b')
ax[1].set_title('Distribution of Transaction Time', fontsize=14)
ax[1].set_xlim([min(time_val), max(time_val)])
# for i =1:30:

plt.show()

#%% [markdown]
# ### distribution of amount with Class:

#%%
counts = df.Class.value_counts()
normal = counts[0]
fraudulent = counts[1]
plt.figure(figsize=(8,6))
sns.barplot(x=counts.index, y=counts)
plt.title('Count of Fraudulent vs. Non-Fraudulent Transactions')
plt.ylabel('Count')
plt.xlabel('Class (0:Non-Fraudulent, 1:Fraudulent)')


#%%
# Class - Amount Plot
plt.subplot(121)
ax = sns.boxplot(x ="Class",y="Amount",
                 data=df)
ax.set_title("Class x Amount", fontsize=20)
ax.set_xlabel("Is Fraud?", fontsize=16)
ax.set_ylabel("Amount", fontsize = 16)
# Total Data Objects with Class 0: 2,84,315 (99.83%) - non-fraud transactions
# Total Data Objects with Class 1: 492 (0.17%) - fraud transactions
#Therefore, the dataset has a strong imbalanced nature, where the problem is two-class classification.

#%% [markdown]
# There are __only 7__ points out of 2.8 Lakh having Amount > 10,000.  
# Therefore these values should be excluded from dataset.

#%%
df[df.Amount > 10000]


#%%
df = df[df.Amount < 10000]
df.describe()


#%%
#New distribution of amount with Class:
plt.subplot(121)
ax = sns.boxplot(x ="Class",y="Amount",
                 data=df)
ax.set_title("Class x Amount", fontsize=20)
ax.set_xlabel("Is Fraud?", fontsize=16)
ax.set_ylabel("Amount", fontsize = 16)

#%% [markdown]
# ### Creating new columns for ease in visualization

#%%

data_new = df
timedelta = pd.to_timedelta(data_new['Time'], unit='s')
#new variable for further analysis
data_new['Time_min'] = (timedelta.dt.components.minutes).astype(int)
#new variable for further analysis
data_new['Time_hour'] = (timedelta.dt.components.hours).astype(int)

#%% [markdown]
# ### Looking at the Amount and time distribuition of FRAUD transactions

#%%

ax = sns.lmplot(y="Amount", x="Time_min", fit_reg=False, aspect=2.5, data=data_new, hue='Class')
plt.title("Amounts by Minutes of Frauds and Normal Transactions",fontsize=12)
#plt.savefig('Amount_VS_Time_Scatter.png')

#%% [markdown]
# ### Exploring the distribuition by Class types throught hours
#%% [markdown]
# Tried to get an idea of trend by hours in order to identify if at certains hours transcations peak, hinting at a higher probability of fraudulent transaction

#%%

plt.figure(figsize=(15,8))
# Non-Fraudulent Transactions over Time (in hr) - GREEN
sns.distplot(data_new[data_new['Class'] == 0]["Time_hour"],
             color='g')
# Fraudulent Transactions over time (in hr) - RED
sns.distplot(data_new[data_new['Class'] == 1]["Time_hour"],
             color='r')
plt.title('Fraud x Normal Transactions by Hours (Red: Fraud; Green:Normal)', fontsize=12)
plt.xlim([-1,25])
#plt.savefig('Time_distribution_fraud_NonFraud.png')

#%% [markdown]
# ### distribution of each class for syntethic variables between V1-V28
#%% [markdown]
# If peaks of both classes occur at different peaks, we can decide on a threshold for that attribute - less data points so bias should be high.

#%%


plt.figure(figsize=(16,28*4))
gs = gridspec.GridSpec(28, 1)
for i, cn in enumerate(data_new[data_new.iloc[:, 1:29].columns]):
    ax = plt.subplot(gs[i])
    sns.distplot(data_new[cn][data_new.Class == 1], bins=50)
    sns.distplot(data_new[cn][data_new.Class == 0], bins=50)
    ax.set_xlabel('')
    ax.set_title('feature: ' + str(cn))
#plt.savefig('Attribute_Distribution_with Class_Seperation.png')

#%% [markdown]
# Which features follow normal distribution , only V14 so we couldn't use properties of Normal distribution -Kush

#%%
from scipy.stats import norm

f, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20, 6))

v14_fraud_dist = df['V14'].loc[df['Class'] == 1].values
sns.distplot(v14_fraud_dist,ax=ax1, fit=norm, color='#FB8861')
ax1.set_title('V14 Distribution \n (Fraud Transactions)', fontsize=14)

v12_fraud_dist = df['V12'].loc[df['Class'] == 1].values
sns.distplot(v12_fraud_dist,ax=ax2, fit=norm, color='#56F9BB')
ax2.set_title('V12 Distribution \n (Fraud Transactions)', fontsize=14)


v10_fraud_dist = df['V10'].loc[df['Class'] == 1].values
sns.distplot(v10_fraud_dist,ax=ax3, fit=norm, color='#C5B3F9')
ax3.set_title('V10 Distribution \n (Fraud Transactions)', fontsize=14)

plt.show()

#%% [markdown]
# As there is __no nominal attribute__, we don't use bar graphs for analysis.   
# __NO__ data cubing performed, as attributes have no hierarchy, and are continuous  
# Scatter plot of all attributes Pairwise (Pair Plots)  

#%%
# sns.set(style="ticks")
# sns.pairplot(df, hue="Class")
#plt.savefig('pairplots_coloured')
# Observation: Fraudulent cases are concentrated near 0 in attribute V20, V27, V28 and Amount.

#%% [markdown]
# ### Box plots

#%%
l = df.columns.values 
number_of_columns= 15
number_of_rows = len(l)-1/number_of_columns #one column is of class so we won't take it for the corelation analysis
plt.figure(figsize=(number_of_columns,5*number_of_rows))
for i in range(0,len(l)-1):
    plt.subplot(number_of_rows + 1,number_of_columns,i+1)
    sns.set_style('whitegrid')
    sns.boxplot(df[l[i]],color='green',orient='v',width=20).set_title(l[i],fontsize=14)
#     plt.tight_layout()

#%% [markdown]
# Deciding on threshold + detecting corrrelation which made us plot correlation heat map
#%% [markdown]
# ### Correlation matrix heat-map
#%% [markdown]
# Most of the pixels are dark pink in colour, which means most of attributes are independent of each other.  
# Some cases are of positive correlation and some are negatively correlated.  
# But Pearson’s product coefficient of all lies between (-0.5 to +0.5). Hence, we are not removing any attribute in this step.  
# Negative correlation with class: V10, V12, V14, V17  
# __We have to make sure we use the subsample in our correlation matrix or else our correlation matrix will be affected by the high imbalance between our classes. This occurs due to the high class imbalance in the original dataframe.__

#%%

plt.subplots(figsize=(20,13 ))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=False)
#plt.savefig('corelation_matrix_heatmap.png')

#%% [markdown]
# ### Checking Missing Values

#%%

print('Are there any Missing values? : ',df.isnull().any().any())

#%% [markdown]
# ###  Data Consistency
#%% [markdown]
# Data outliers are indicative fraud transactions - so explicit data smoothing not performed.   
# Extreme outliers will be removed while training the model.
# We also plan to remove outliers during training the model via a new technique.
# 
# Resolving Inconsistencies:  
# As V1 to V28 are Principal Components, they don't have any inconsistencies.  
# 
#%% [markdown]
# ### Scaling Data using Robust Scalar

#%%
# Reason: robust scaler is immune to outliers, as median is chosen as the central tendancy.
from sklearn.preprocessing import StandardScaler, RobustScaler

rob_scaler = RobustScaler()

df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))

df.drop(['Time','Amount'], axis=1, inplace=True)
df = df[['scaled_time','scaled_amount', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
       'Class']]
df.to_csv("scaled_data.csv")
print('Scaled Data\n')
df.head(10)

#%% [markdown]
# ### Splitting Data 

#%%

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


# print('No Frauds', round(df['Class'].value_counts()[0]/len(df) * 100,2), '% of the dataset')
# print('Frauds', round(df['Class'].value_counts()[1]/len(df) * 100,2), '% of the dataset')

X = df.drop('Class', axis=1)
y = df['Class']

sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

for train_index, test_index in sss.split(X, y):
    # print("Train:", train_index, "Test:", test_index)
    original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]

# We already have X_train and y_train for undersample data thats why I am using original to distinguish and to not overwrite these variables.
# original_Xtrain, original_Xtest, original_ytrain, original_ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the Distribution of the labels
original_Xtrain.to_csv("X_train.csv")
original_ytrain.to_csv("y_train.csv")
original_Xtest.to_csv("X_test.csv")
original_ytest.to_csv("y_test.csv")

# Turn into an array
original_Xtrain = original_Xtrain.values
original_Xtest = original_Xtest.values
original_ytrain = original_ytrain.values
original_ytest = original_ytest.values

# See if both the train and test label distribution are similarly distributed
train_unique_label, train_counts_label = np.unique(original_ytrain, return_counts=True)
test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)
print('-' * 100)

print('Label Distributions: \n')
print(train_counts_label/ len(original_ytrain))
print(test_counts_label/ len(original_ytest))

#%% [markdown]
# ### Reducing Rows via Random-Under Sampling: Numerosity Reduction:  
# 
# Since our classes are highly skewed we should make them equivalent in order to have a normal distribution of the classes.  
# 
# Lets shuffle the data before creating the subsamples  
# 
# Cosine Similarity Analysis not performed as data has very few zeros.  
# Parametric Methods for numerosity reduction- NOT Applicable as we need to detect outliers  

#%%


df = df.sample(frac=1)

# amount of fraud classes 492 rows.
fraud_df = df.loc[df['Class'] == 1]
non_fraud_df = df.loc[df['Class'] == 0][:492]

normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

# Shuffle dataframe rows
new_df = normal_distributed_df.sample(frac=1, random_state=42)
new_df.to_csv('new_data.csv')
new_df.head()

#%% [markdown]
# ## Further Analysis and Preprocessing of new Balanced Data Frame

#%%

print('Distribution of the Classes in the subsample dataset')
print(new_df['Class'].value_counts()/len(new_df))

sns.countplot('Class', data=new_df)
plt.title('Equally Distributed Classes', fontsize=14)
plt.show()

#%% [markdown]
# Now with this Balanced Dataset, we can apply more Supervised Learning Algorithms which was a short-coming on previous dataset (due to high class imbalance).
#%% [markdown]
# New Boxplot

#%%

l = new_df.columns.values 
number_of_columns= 15
number_of_rows = len(l)-1/number_of_columns #one column is of class so we won't take it for the corelation analysis
plt.figure(figsize=(number_of_columns,5*number_of_rows))
for i in range(0,len(l)-1):
    plt.subplot(number_of_rows + 1,number_of_columns,i+1)
    #sns.set_style('whitegrid')
    sns.boxplot(new_df[l[i]],color='blue',orient='v',width=20).set_title(l[i],fontsize=14)
#     plt.tight_layout()

#plt.savefig('boxplots_new.png')

#%% [markdown]
# ### Correlation matrix heat-map on new Balanced Data

#%%


plt.subplots(figsize=(20,13 ))
correlation_matrix = new_df.corr()
sns.heatmap(correlation_matrix, annot=False)
#plt.savefig('corelation_matrix_heatmap_new.png')


#%%
l = new_df.columns.values
for i in range(31):
    for j in range(i):
        if(abs(correlation_matrix.iloc[i,j])>0.9 and i != j):
            print(l[i],l[j])

#%% [markdown]
# ### Note: _We are not dropping any of the correlated columns because after performing classification, we found that the accuracy of our model decreases incase of dropping the columns_.

#%%
# new_df = new_df.drop(['V12', 'V17', 'V18'], axis = 1)


#%%
new_df.head()


#%%
new_df.describe()

#%% [markdown]
# The preprocessed dataset has __30__ features and ~1000 data points with an equiprobable distribution 
# 
#%% [markdown]
# Conclusion  
# 0) heavy imbalance  
# 1) Reducing data points from x to y  
# 2) Time and amount only explainable attributes - using prior knowldege for that (increasing bias necessary)  
# 2.5) 10k outlier  
# 3) Identifying properties of PCA features  
# 4) New correlations after under-sampling -dropping features 12,17,18
# 5) 2 way sampling and future tasks  
# 6) t-SNE  
#%% [markdown]
# ### --------------------------------------------------------------------------------------------------------------------------------
# ### --------------------------------------------------------------------------------------------------------------------------------
#%% [markdown]
# ### Outlier Removal

#%%
# # -----> V14 Removing Outliers (Highest Negative Correlated with Labels)
v14_fraud = new_df['V14'].loc[new_df['Class'] == 1].values
q25, q75 = np.percentile(v14_fraud, 25), np.percentile(v14_fraud, 75)
print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))
v14_iqr = q75 - q25
print('iqr: {}'.format(v14_iqr))

v14_cut_off = v14_iqr * 1.5
v14_lower, v14_upper = q25 - v14_cut_off, q75 + v14_cut_off
print('Cut Off: {}'.format(v14_cut_off))
print('V14 Lower: {}'.format(v14_lower))
print('V14 Upper: {}'.format(v14_upper))

outliers = [x for x in v14_fraud if x < v14_lower or x > v14_upper]
print('Feature V14 Outliers for Fraud Cases: {}'.format(len(outliers)))
print('V14 outliers:{}'.format(outliers))

new_df = new_df.drop(new_df[(new_df['V14'] > v14_upper) | (new_df['V14'] < v14_lower)].index)
print('----' * 44)

# -----> V12 removing outliers from fraud transactions
v12_fraud = new_df['V12'].loc[new_df['Class'] == 1].values
q25, q75 = np.percentile(v12_fraud, 25), np.percentile(v12_fraud, 75)
v12_iqr = q75 - q25

v12_cut_off = v12_iqr * 1.5
v12_lower, v12_upper = q25 - v12_cut_off, q75 + v12_cut_off
print('V12 Lower: {}'.format(v12_lower))
print('V12 Upper: {}'.format(v12_upper))
outliers = [x for x in v12_fraud if x < v12_lower or x > v12_upper]
print('V12 outliers: {}'.format(outliers))
print('Feature V12 Outliers for Fraud Cases: {}'.format(len(outliers)))
new_df = new_df.drop(new_df[(new_df['V12'] > v12_upper) | (new_df['V12'] < v12_lower)].index)
print('Number of Instances after outliers removal: {}'.format(len(new_df)))
print('----' * 44)


# Removing outliers V10 Feature
v10_fraud = new_df['V10'].loc[new_df['Class'] == 1].values
q25, q75 = np.percentile(v10_fraud, 25), np.percentile(v10_fraud, 75)
v10_iqr = q75 - q25

v10_cut_off = v10_iqr * 1.5
v10_lower, v10_upper = q25 - v10_cut_off, q75 + v10_cut_off
print('V10 Lower: {}'.format(v10_lower))
print('V10 Upper: {}'.format(v10_upper))
outliers = [x for x in v10_fraud if x < v10_lower or x > v10_upper]
print('V10 outliers: {}'.format(outliers))
print('Feature V10 Outliers for Fraud Cases: {}'.format(len(outliers)))
new_df = new_df.drop(new_df[(new_df['V10'] > v10_upper) | (new_df['V10'] < v10_lower)].index)
print('Number of Instances after outliers removal: {}'.format(len(new_df)))


#%%
f,(ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,6))

colors = ['#B3F9C5', '#f9c5b3']
# Boxplots with outliers removed
# Feature V14
sns.boxplot(x="Class", y="V14", data=new_df,ax=ax1, palette=colors)
ax1.set_title("V14 Feature \n Reduction of outliers", fontsize=14)
ax1.annotate('Fewer extreme \n outliers', xy=(0.98, -17.5), xytext=(0, -12),
            arrowprops=dict(facecolor='black'),
            fontsize=14)

# Feature 12
sns.boxplot(x="Class", y="V12", data=new_df, ax=ax2, palette=colors)
ax2.set_title("V12 Feature \n Reduction of outliers", fontsize=14)
ax2.annotate('Fewer extreme \n outliers', xy=(0.98, -17.3), xytext=(0, -12),
            arrowprops=dict(facecolor='black'),
            fontsize=14)

# Feature V10
sns.boxplot(x="Class", y="V10", data=new_df, ax=ax3, palette=colors)
ax3.set_title("V10 Feature \n Reduction of outliers", fontsize=14)
ax3.annotate('Fewer extreme \n outliers', xy=(0.95, -16.5), xytext=(0, -12),
            arrowprops=dict(facecolor='black'),
            fontsize=14)


plt.show()


#%%
new_df.head()


#%%
new_df.to_csv('outlier_removed_df.csv')


#%%


#%% [markdown]
# ## Classification algorithms that we will be using:  
# 
# ### Logistic Regression
# ### K Nearest Neighbours
# ### Linear Discriminant Analysis
# ### Classification Trees
# ### Support Vector Classifier
# ### Random Forest Classifier
# ### XGBoost Classifier   

#%%



