#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Supress unnecessary warnings

import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Importing the NumPy and Pandas packages

import numpy as np
import pandas as pd

#import visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

#import stats library
from scipy import stats
import statsmodels.api as sm

#import sklearn libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import metrics
from sklearn.metrics import classification_report,recall_score,roc_auc_score,roc_curve,accuracy_score,precision_score,precision_recall_curve,confusion_matrix
from sklearn.preprocessing import LabelEncoder

#import miscellaneous libraries
pd.set_option("display.max_columns",None)
pd.set_option("display.max_colwidth",200)


# ### Importing the "Leads" Dataset

# In[3]:


# Read the dataset
leads = pd.read_csv(r"D:\op downlond\Lead+Scoring+Case+Study\Lead Scoring Assignment\Leads.csv")


# In[4]:


leads.head()


# In[5]:


#Checking the Shape of dataset
leads.shape


# In[6]:


# Inspecting the different columns in the dataset

leads.columns


# In[7]:


# Checking the summary of the dataset
leads.describe()


# In[8]:


# Checking the info to see the types of the feature variables and the null values present
leads.info()


# As it seems that there are quite a few categorical variables present in this dataset for which we will need to create dummy variables. Also, there are a lot of null values present as well, so we will need to treat them accordingly.

# ## Step 1: Data Cleaning and Preparation

# In[9]:


# Checking the number of missing values in each column
leads.isnull().sum().sort_values(ascending=False)


# As it is  clearly seen there are a lot of columns which have high number of missing values. Clearly, these columns are not useful. Since, there are 9000 datapoints in our dataframe, let's eliminate the columns having greater than 3000 missing values as they are of no use to us.

# In[10]:


# Droping all the columns in which greater than 
for c in leads.columns:
    if leads[c].isnull().sum()>3000:
        leads.drop(c, axis=1,inplace=True)


# In[11]:


leads.isnull().sum().sort_values(ascending=False)


# In[12]:


#checking value counts of "City" column
leads['City'].value_counts(dropna=False)


# `Mumbai` has highest numbers of leads

# As you might be able to interpret, the variable `City` won't be of any use in our analysis. So it's best that we drop it.

# In[13]:


# dropping the "City" feature
leads.drop(['City'], axis = 1, inplace = True)


# In[14]:


#checking value counts of "Country" column
leads['Country'].value_counts(dropna=False)


# Highest number of leads from `INDIA`

# In[15]:


# dropping the "Country" feature
leads.drop(['Country'], axis = 1, inplace = True)


# In[16]:


#Now checking the percentage of missing values in each column

round(100*(leads.isnull().sum()/len(leads.index)), 2)


# In[17]:


# Checking the number of null values again
leads.isnull().sum().sort_values(ascending=False)


# ### Visualizing the features with `Select` values

# In[18]:


def countplot(x, fig):
    plt.subplot(2,2, fig)
    sns.countplot(leads[x])
    plt.title('Count across'+' '+ x, size = 16)
    plt.xlabel(x,size = 14)
    plt.xticks(rotation = 90)

plt.figure(figsize=(15,10))

countplot('How did you hear about X Education',1)
countplot('Lead Profile',2)
countplot('Specialization',3)



plt.tight_layout()


# there are a few columns in which there is a level called `'Select'` which basically means that the student had not selected the option for that particular column which is why it shows 'Select'. These values are as good as missing values and hence we need to identify the value counts of the level 'Select' in all the columns that it is present.

# In[19]:


# checking the value counts of all the columns

for c in leads:
    print(leads[c].astype('category').value_counts())
    print('___________________________________________________')


# The following three columns now have the level 'Select'. Let's check them once again.

# In[20]:


leads['Lead Profile'].astype('category').value_counts()


# In[21]:


leads['How did you hear about X Education'].value_counts()


# In[22]:


leads['Specialization'].value_counts()


# ### Visualizing the features

# In[23]:


def countplot(x, fig):
    plt.subplot(4,2, fig)
    sns.countplot(leads[x])
    plt.title('Count across'+' '+ x, size = 16)
    plt.xlabel(x,size = 14)
    plt.xticks(rotation = 90)

plt.figure(figsize=(18,25))


countplot('What matters most to you in choosing a course',1)
countplot('What is your current occupation',2)
countplot('Specialization',3)

plt.tight_layout()


# As it can be seen that the levels of `"Lead Profile"` and `"How did you hear about X Education"` have a lot of rows which have the value Select which is of no use to the analysis
# 
# So it's best that we drop them.

# In[24]:


# dropping Lead Profile and How did you hear about X Education cols
leads.drop(['Lead Profile', 'How did you hear about X Education'], axis = 1, inplace = True)


# Also we notice that, when we got the value counts of all the columns, there were a few columns in which only one value was majorly present for all the data points. These include `Do Not Call`, `Search`, `Magazine`, `Newspaper Article`, `X Education Forums`, `Newspaper`, `Digital Advertisement`, `Through Recommendations`, `Receive More Updates About Our Courses`, `Update me on Supply Chain Content`, `Get updates on DM Content`, `I agree to pay the amount through cheque`. Since practically all of the values for these variables are `No`, it's best that we drop these columns as they won't help with our analysis.

# In[25]:


from matplotlib import pyplot as plt
import seaborn as sns
sns.pairplot(leads,diag_kind='kde',hue='Converted')
plt.show()


# In[26]:


x_edu = leads[['TotalVisits','Total Time Spent on Website','Page Views Per Visit','Converted']]
sns.pairplot(x_edu,diag_kind='kde',hue='Converted')
plt.show()


# In[27]:


from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer()
transformedx_edu = pd.DataFrame(pt.fit_transform(x_edu))
transformedx_edu.columns = x_edu.columns
transformedx_edu.head()


# In[28]:


sns.pairplot(transformedx_edu,diag_kind='kde',hue='Converted')
plt.show()


# In[29]:


# Dropping the above columns


# In[30]:


leads.drop(['Do Not Call', 'Search', 'Magazine', 'Newspaper Article', 'X Education Forums', 'Newspaper', 
            'Digital Advertisement', 'Through Recommendations', 'Receive More Updates About Our Courses', 
            'Update me on Supply Chain Content', 'Get updates on DM Content', 
            'I agree to pay the amount through cheque'], axis = 1, inplace = True)


# In[31]:


leads['What matters most to you in choosing a course'].value_counts()


# The variable `What matters most to you in choosing a course` has the `level Better Career Prospects` 6528 times while the other two levels appear once twice and once respectively. 
# 
# So we should dropping this column as well.

# In[32]:


leads.drop(['What matters most to you in choosing a course'], axis = 1, inplace=True)


# In[33]:


# Checking the number of null values again
leads.isnull().sum().sort_values(ascending=False)


# Now, there's the column `What is your current occupation` which has a lot of null values. Now you can drop the entire row but since we have already lost so many feature variables, we choose not to drop it as it might turn out to be significant in the analysis. So let's just drop the null rows for the column `What is you current occupation`.

# In[34]:


# Dropping the null values rows in the column 'What is your current occupation'

leads = leads[~pd.isnull(leads['What is your current occupation'])]


# In[35]:


# Observing Correlation
# figure size
plt.figure(figsize=(10,8))

# heatmap
sns.heatmap(leads.corr(), annot=True,cmap="BrBG", robust=True,linewidth=0.1, vmin=-1 )
plt.show()


# ### Analysing Categorical features

# In[36]:


conv = leads.select_dtypes(include ="object").columns
for i in conv:
    
    plt.figure(figsize =(15,5))
    sns.countplot(leads[i], hue=leads.Converted)
    plt.xticks(rotation = 90)
    plt.title('Target variable in'+' '+ i)
    plt.xlabel(i)
    plt.show()


# In[37]:


# Checking the number of null values again
leads.isnull().sum().sort_values(ascending=False)


# Since now the number of null values present in the columns are quite small we can simply drop the rows in which these null values are present.

# In[38]:


# Dropping the null values rows in the column 'TotalVisits'

leads = leads[~pd.isnull(leads['TotalVisits'])]


# In[39]:


# Checking the number of null values again
leads.isnull().sum().sort_values(ascending=False)


# In[40]:


# Dropping the null values rows in the column 'Lead Source'

leads = leads[~pd.isnull(leads['Lead Source'])]


# In[41]:


# Checking the number of null values again
leads.isnull().sum().sort_values(ascending=False)


# In[42]:


# Drop the null values rows in the column 'Specialization'

leads = leads[~pd.isnull(leads['Specialization'])]


# In[43]:


# Checking the number of null values again
leads.isnull().sum().sort_values(ascending=False)


# Now your data doesn't have any null values. Let's now check the percentage of rows that we have retained.

# In[44]:


print(len(leads.index))
print(len(leads.index)/9240)


# We still have around 69% of the rows which seems good enough.

# In[45]:


# Let's look at the dataset again

leads.head()


# Now, clearly the variables `Prospect ID` and `Lead Number` won't be of any use in the analysis, so it's best that we drop these two variables.

# In[46]:


# Dropping the "Prospect ID" and "Lead Number" 
leads.drop(['Prospect ID', 'Lead Number'], 1, inplace = True)


# In[47]:


leads.head()


# ### Dummy variable creation
# 
# The next step is to dealing with the categorical variables present in the dataset. So first take a look at which variables are actually categorical variables.

# In[48]:


# Checking the columns which are of type 'object'

temp = leads.loc[:, leads.dtypes == 'object']
temp.columns


# In[49]:


# Demo Cell
df = pd.DataFrame({'P': ['p', 'q', 'p']})
df


# In[50]:


pd.get_dummies(df)


# In[51]:


pd.get_dummies(df, prefix=['col1'])


# In[52]:


# Creating dummy variables using the 'get_dummies' command
dummy = pd.get_dummies(leads[['Lead Origin', 'Lead Source', 'Do Not Email', 'Last Activity',
                              'What is your current occupation','A free copy of Mastering The Interview', 
                              'Last Notable Activity']], drop_first=True)

# Add the results to the master dataframe
leads = pd.concat([leads, dummy], axis=1)


# In[53]:


# Creating dummy variable separately for the variable 'Specialization' since it has the level 'Select' 
# which is useless so we
# drop that level by specifying it explicitly

dummy_spl = pd.get_dummies(leads['Specialization'], prefix = 'Specialization')
dummy_spl = dummy_spl.drop(['Specialization_Select'], 1)
leads = pd.concat([leads, dummy_spl], axis = 1)


# In[54]:


# Dropping the variables for which the dummy variables have been created

leads = leads.drop(['Lead Origin', 'Lead Source', 'Do Not Email', 'Last Activity',
                   'Specialization', 'What is your current occupation',
                   'A free copy of Mastering The Interview', 'Last Notable Activity'], 1)


# In[55]:


# Let's take a look at the dataset again

leads.head()


# ### Test-Train Split
# 
# The next step is to spliting the dataset into training an testing sets.

# In[56]:


# Importing the `train_test_split` library


# In[57]:


# Put all the feature variables in X

X = leads.drop(['Converted'], 1)
X.head()


# In[58]:


y = leads['Converted']

y.head()


# In[59]:


# Spliting the dataset into 70% train and 30% test

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# In[60]:


#lets check the shape
print("X_train Size", X_train.shape)
print("y_train Size", y_train.shape)


# ### Scaling
# 
# Now there are a few numeric variables present in the dataset which have different scales. So let's go ahead and scale these variables.

# In[61]:


# Importing the 'MinMax scaler' Library


# In[62]:


# Scaling the three numeric features present in the dataset

scaler = MinMaxScaler()

X_train[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']] = scaler.fit_transform(X_train[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']])

X_train.head()


# ### Looking at the correlations
# 
# Let's now look at the correlations. Since the number of variables are pretty high, it's better that we look at the table instead of plotting a heatmap

# In[63]:


# Looking at the correlation table
plt.figure(figsize = (25,15))
sns.heatmap(leads.corr())
plt.show()


# ## Step 2: Model Building
# 
# Let's now move to model building. As you can see that there are a lot of variables present in the dataset which we cannot deal with. So the best way to approach this is to select a small set of features from this pool of variables using RFE.

# In[64]:


# Importing the 'LogisticRegression' and creating a LogisticRegression object
logreg = LogisticRegression()


# In[66]:


# Importing the 'RFE' and select 15 variables

rfe = RFE(logreg, n_features_to_select=15)

# Fit the RFE model on the training data
rfe.fit(X_train, y_train)


# In[67]:


# Let's take a look at which features have been selected by RFE

list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[68]:


# Putting all the columns selected by RFE in the variable 'col'

col = X_train.columns[rfe.support_]


# Now we have all the variables selected by RFE and since we care about the statistics part, i.e. the p-values and the VIFs, let's use these variables to create a logistic regression model using statsmodels.

# In[69]:


# Select only the columns selected by RFE

X_train = X_train[col]


# In[70]:


# Importing 'statsmodels'


# ### Model 1

# In[71]:


# Fit a logistic Regression model on X_train after adding a constant and output the summary

X_train_sm = sm.add_constant(X_train)
logm2 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# There are quite a few variable which have a p-value greater than 0.05. We will need to take care of them. But first, let's also look at the VIFs.

# ### Checking `VIF`

# In[72]:


# Importing the 'variance_inflation_factor' library


# In[73]:


# Make a VIF dataframe for all the variables present

vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# VIFs seem to be in a decent range except for three variables.
# 
# Let's first drop the variable `Lead Source_Reference` since it has a high p-value as well as a high VIF.

# In[74]:


X_train.drop('Lead Source_Reference', axis = 1, inplace = True)


# ### Model 2

# In[75]:


# Refit the model with the new set of features

logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()


# #### Checking VIF

# In[76]:


# Make a VIF dataframe for all the variables present

vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# The VIFs are now all less than 5. So let's drop the ones with the high p-values beginning with `Last Notable Activity_Had a Phone Conversation`.

# In[77]:


X_train.drop('Last Notable Activity_Had a Phone Conversation', axis = 1, inplace = True)


# ### Model 3

# In[78]:


# Refit the model with the new set of features

logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()


# Dropping the `What is your current occupation_Housewife` as having high P value

# In[79]:


X_train.drop('What is your current occupation_Housewife', axis = 1, inplace = True)


# ### Model 4

# In[80]:


# Refit the model with the new set of features

logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()


# Droppint hre  `What is your current occupation_Working Professional` as having high P value

# In[81]:


X_train.drop('What is your current occupation_Working Professional', axis = 1, inplace = True)


# ### Model 4

# In[82]:


# Refit the model with the new set of features

logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
res = logm1.fit()
res.summary()


# #### Checking final VIF

# In[83]:


# Making a VIF dataframe for all the variables present

vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# ## Step 3: Model Evaluation
# 
# Now, both the p-values and VIFs seem decent enough for all the variables. So let's go ahead and make predictions using this final set of features.

# In[84]:


# Use 'predict' to predict the probabilities on the train set

y_train_pred = res.predict(sm.add_constant(X_train))
y_train_pred[:10]


# In[85]:


# Reshaping it into an array

y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# ### Creating a dataframe with the actual conversion flag and the predicted probabilities

# In[86]:


# Creating a new dataframe containing the actual conversion flag and the probabilities predicted by the model

y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Conversion_Prob':y_train_pred})
y_train_pred_final.head()


# ### Creating new column 'Predicted' with 1 if Paid_Prob > 0.5 else 0

# In[87]:


y_train_pred_final['Predicted'] = y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.5 else 0)

# Let's see the head
y_train_pred_final.head()


# Now that you have the probabilities and have also made conversion predictions using them, it's time to evaluate the model.

# In[88]:


# Importing the 'metrics' library from sklearn for evaluation


# ### Creating the `Confusion matrix`
# 

# In[89]:


confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted )
print(confusion)


# In[90]:


# Let's check the overall accuracy

print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Predicted))


# In[91]:


# Let's evaluate the other metrics as well

TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[92]:


# Calculating the 'sensitivity'

TP/(TP+FN)


# In[93]:


# Calculating the 'specificity'

TN/(TN+FP)


# ### Finding the Optimal Cutoff
# Now 0.5 was just arbitrary to loosely check the model performace. But in order to get good results, you need to optimise the threshold. So first let's plot an ROC curve to see what AUC we get.

# In[94]:


# ROC function

def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[95]:


fpr, tpr, thresholds = metrics.roc_curve(y_train_pred_final.Converted,
                    y_train_pred_final.Conversion_Prob, 
                                         drop_intermediate=False)


# In[96]:


# Importing the 'matplotlib'  to plot the ROC curve`


# In[97]:


# Calling the ROC function

draw_roc(y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob)


# The area under the curve of the ROC is 0.86 which is quite good. So we seem to have a good model. Let's also check the sensitivity and specificity tradeoff to find the optimal cutoff point.

# In[98]:


# Let's create columns with different probability cutoffs 

numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[99]:


# Let's create a dataframe to see the values of accuracy, sensitivity, and specificity at 
# different values of probabiity cutoffs

cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# In[100]:


# Let's plot it as well

cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# As you can see that around `0.42`, you get the optimal values of the three metrics. So let's choose 0.42 as our cutoff now.

# In[101]:


y_train_pred_final['final_predicted'] = y_train_pred_final.Conversion_Prob.map( lambda x: 1 if x > 0.42 else 0)

y_train_pred_final.head()


# In[102]:


# Let's checking the `accuracy` now

metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)


# In[103]:


# Let's create the confusion matrix once again

confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted )
confusion2


# In[104]:


# Let's evaluate the other metrics as well

TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[105]:


# Calculating the 'Sensitivity'

TP/(TP+FN)


# In[106]:


# Calculating the 'Specificity'

TN/(TN+FP)


# This cutoff point seems good to go!

# ## Step 4: Making Predictions on the Test Set
# Let's now make predicitons on the test set

# In[107]:


# Scaling the test set as well using just 'transform'

X_test[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']] =  scaler.transform(X_test[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']])


# In[108]:


# Selecting the columns in X_train for X_test as well

X_test = X_test[col]
X_test.head()


# In[109]:


# Adding a constant to X_test

X_test_sm = sm.add_constant(X_test[col])


# In[110]:


# Checking X_test_sm

X_test_sm


# In[111]:


# Dropping the required columns from X_test as well

X_test.drop(['Lead Source_Reference', 'What is your current occupation_Housewife', 
             'What is your current occupation_Working Professional', 
                     'Last Notable Activity_Had a Phone Conversation'], 1, 
                                inplace = True)


# In[112]:


# Make predictions on the test set and store it in the variable 'y_test_pred'

y_test_pred = res.predict(sm.add_constant(X_test))


# In[113]:


y_test_pred[:10]


# In[114]:


# Converting y_pred to a dataframe

y_pred_1 = pd.DataFrame(y_test_pred)


# In[115]:


# Let's see the head

y_pred_1.head()


# In[116]:


# Converting y_test to dataframe

y_test_df = pd.DataFrame(y_test)


# In[117]:


# Remove index for both dataframes to append them side by side 

y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[118]:


# Append y_test_df and y_pred_1

y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)


# In[119]:


# Check 'y_pred_final'

y_pred_final.head()


# In[120]:


# Rename the column 

y_pred_final= y_pred_final.rename(columns = {0 : 'Conversion_Prob'})


# In[121]:


# Let's see the head of y_pred_final

y_pred_final.head()


# In[122]:


# Make predictions on the test set using 0.45 as the cutoff

y_pred_final['final_predicted'] = y_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.42 else 0)


# In[123]:


# Check y_pred_final

y_pred_final.head()


# In[124]:


# Let's check the overall accuracy

metrics.accuracy_score(y_pred_final['Converted'], y_pred_final.final_predicted)


# In[125]:


confusion2 = metrics.confusion_matrix(y_pred_final['Converted'], y_pred_final.final_predicted )
confusion2


# In[126]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[127]:


# Calculating the 'sensitivity'
TP / float(TP+FN)


# In[128]:


# Calculating the 'specificity'
TN / float(TN+FP)


# ### Precision-Recall View
# Let's now also build the training model using the precision-recall view

# In[129]:


#Looking at the confusion matrix again


# In[130]:


confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted )
confusion


# #### Precision = 
#          TP / TP + FP

# In[131]:


confusion[1,1]/(confusion[0,1]+confusion[1,1])


# #### Recall = 
#           TP / TP + FN

# In[132]:


confusion[1,1]/(confusion[1,0]+confusion[1,1])


# ### Precision and recall tradeoff

# Importing the `Precision recall curve` library

# In[133]:


y_train_pred_final.Converted, y_train_pred_final.Predicted


# In[134]:


p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob)


# In[135]:


plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# In[136]:


y_train_pred_final['final_predicted'] = y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.44 else 0)

y_train_pred_final.head()


# In[137]:


# Let's checking the `accuracy` now

metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)


# In[138]:


# Let's creating the confusion matrix once again

confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted )
confusion2


# In[139]:


# Let's evaluate the other metrics as well

TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# ### Precision

# In[140]:


TP/(TP+FP)


# ### Recall

# In[141]:


TP/(TP+FN)


# This cutoff point seems good to go!

# ## Step 5: Making Predictions on the Test Set
# Let's now make predicitons on the test set.

# In[142]:


# Making predictions on the test set and store it in the variable 'y_test_pred'

y_test_pred = res.predict(sm.add_constant(X_test))


# In[143]:


y_test_pred[:10]


# In[144]:


# Converting y_pred to a dataframe

y_pred_1 = pd.DataFrame(y_test_pred)


# In[145]:


# Let's see the head

y_pred_1.head()


# In[146]:


# Converting y_test to dataframe

y_test_df = pd.DataFrame(y_test)


# In[147]:


# Removing index for both dataframes to append them side by side 

y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[148]:


# Append y_test_df and y_pred_1

y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)


# In[149]:


# Checking the 'y_pred_final'

y_pred_final.head()


# In[150]:


# Rename the column 

y_pred_final= y_pred_final.rename(columns = {0 : 'Conversion_Prob'})


# In[151]:


# Let's see the head of y_pred_final

y_pred_final.head()


# In[152]:


# Making predictions on the test set using 0.44 as the cutoff

y_pred_final['final_predicted'] = y_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.44 else 0)


# In[153]:


# Checking y_pred_final

y_pred_final.head()


# In[154]:


# Let's checking the overall accuracy

metrics.accuracy_score(y_pred_final['Converted'], y_pred_final.final_predicted)


# In[155]:


confusion2 = metrics.confusion_matrix(y_pred_final['Converted'], y_pred_final.final_predicted )
confusion2


# In[156]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[157]:


# Calculating the Precision

TP/(TP+FP)


# In[158]:


# Calculating Recall

TP/(TP+FN)


# # Conclusion :

# - The logistic regression model predicts the probability of the target variable having a certain value, rather than predicting the value of the target variable directly. Then a cutoff of the probability is used to obtain the predicted value of the target variable.
# - Here, the logistic regression model is used to predict the probabilty of conversion of a customer.
# - Optimum cut off is chosen to be 0.27 i.e. any lead with greater than 0.27 probability of converting is predicted as Hot Lead (customer will convert) and any lead with 0.27 or less probability of converting is predicted as Cold Lead (customer will not convert).
# - Our final Logistic Regression Model is built with 14 features.
# - Features used in final model are 
# 
#     ['Do Not Email', 'Lead Origin_Lead Add Form',
#    'Lead Source_Welingak Website', 'Last Activity_SMS Sent', 'Tags_Busy',
#    'Tags_Closed by Horizzon', 'Tags_Lost to EINS', 'Tags_Ringing',
#    'Tags_Will revert after reading the email', 'Tags_switched off',
#    'Lead Quality_Not Sure', 'Lead Quality_Worst',
#    'Last Notable Activity_Modified',
#    'Last Notable Activity_Olark Chat Conversation']
#    
#  
# - The top three categorical/dummy variables in the final model are ‘Tags_Lost to EINS’, ‘Tags_Closed by Horizzon’, ‘Lead Quality_Worst’ with respect to the absolute value of their coefficient factors.
#  
#  ‘Tags_Lost to EINS’, ‘Tags_Closed by Horizzon’ are obtained by encoding original categorical variable ‘Tags’. ‘Lead Quality_Worst’ is obtained by encoding the categorical variable ‘Lead Quality’.
#  
#  
# - Tags_Lost to EINS (Coefficient factor = 9.578632)
# - Tags_Closed by Horizzon (Coefficient factor = 8.555901)
# - Lead Quality_Worst (Coefficient factor =-3.943680)
# - The final model has Sensitivity of 0.928, this means the model is able to predict 92% customers out of all the converted customers, (Positive conversion) correctly.
# - The final model has Precision of 0.68, this means 68% of predicted hot leads are True Hot Leads.
# - We have also built an reusable code block which will predict Convert value and Lead Score given training, test data and a cut-off. Different cutoffs can be used depending on the use-cases (for eg. when high sensitivity is required, when model have optimum precision score etc.)

# In[ ]:





# In[ ]:




