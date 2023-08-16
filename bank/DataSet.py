#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


bank_df = pd.read_csv('bank.csv')
#bank_df = pd.read_csv('bank_formatted.csv')
bank_df.head(3)


# In[3]:


import pandas as pd
bank_df = pd.read_csv('bank.csv', sep=';', lineterminator='\n')
bank_df.to_csv('bank_test_formatted.csv')


# In[4]:


unknown_count=(bank_df['job']=='unknown').sum() 
unknown_count


# In[5]:


for column in bank_df:
    unknown_count = 0
    unknown_count=(bank_df[column]=='unknown').sum() 
    print(f'{column} {unknown_count}')


# In[6]:


bank_df = bank_df[bank_df['job'] != 'unknown']


# In[7]:


bank_df.to_csv('bank_df_test_filtered.csv')


# In[8]:


bank_df_filtered = pd.read_csv('bank_df_test_filtered.csv')


# In[9]:


bank_df_filtered.columns
bank_df_filtered.head(4)


# In[10]:


count_nodefault= bank_df_filtered[bank_df_filtered['default'] == 'no']
count_nodefault.shape


# In[11]:


bank_df_filtered.describe()


# ### Visualization

# In[12]:


sns.set_style('whitegrid')
bank_df_filtered['age'].hist(bins=30)
plt.xlabel('age')


# In[13]:


sns.jointplot(x='age',y='balance',data=bank_df_filtered)


# In[14]:


#sns.jointplot(x='age',y='duration',data=bank_df_filtered,color='red',kind='kde')
#sns.pairplot(bank_df_filtered,hue='y',palette='bwr')


# #### Logistic Regression

# In[15]:


bank_df_filtered['default'] = bank_df_filtered['default'].map({'yes': 1, 'no': 0})
bank_df_filtered['housing'] = bank_df_filtered['housing'].map({'yes': 1, 'no': 0})
bank_df_filtered['loan'] = bank_df_filtered['loan'].map({'yes': 1, 'no': 0})
bank_df_filtered['y'] = bank_df_filtered['y'].map({'unknown':9,'yes': 1, 'no': 0})  
bank_df_filtered['marital'] = bank_df_filtered['marital'].map({'single':1,'married': 1, 'divorced': 2})  
bank_df_filtered['job'] = bank_df_filtered['job'].map({'management':1,'technician': 2, 'entrepreneur': 3,'blue-collar':3,'retired':4,'admin.':5,'services':6,'self-employed':7,'unemployed':8,'housemaid':9,'student':10 })  
bank_df_filtered['education'] = bank_df_filtered['education'].map({'tertiary':1,'secondary': 2, 'primary': 3,'unknown':9})  
bank_df_filtered['contact'] = bank_df_filtered['contact'].map({'cellular':1,'telephone': 2, 'unknown': 9})  
bank_df_filtered['poutcome'] = bank_df_filtered['poutcome'].map({'failure':1,'success': 2,'other':3, 'unknown': 9})  
bank_df_filtered['month'] = bank_df_filtered['month'].map({'jan':1,'feb': 2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'oct':10,'nov':11,'dec':12})  


# In[16]:


def age_categories():
    age_thresholds=bank_df_filtered['age']
    for age in age_thresholds:
        if age>0 and age <40:
            bank_df_filtered['age_category'] = 1
        elif age>40 and age <60:
            bank_df_filtered['age_category'] = 2
        elif age>60:
            bank_df_filtered['age_category'] = 3
age_categories()


# In[17]:


bank_df_filtered.head(1)


# In[18]:


bank_df_filtered['month'].unique()


# In[19]:


bank_df_filtered.isna().sum()


# In[20]:


from sklearn.model_selection import train_test_split
X = bank_df_filtered.drop('y',axis = 1)
X = bank_df_filtered.drop('month',axis = 1)
#X = bank_df_filtered[['balance',]]
y = bank_df_filtered['y']
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[21]:


from sklearn.linear_model import LogisticRegression
def logistic_regression():
    logmodel = LogisticRegression()
    #logmodel.fit(X_train,y_train)
    logmodel.fit(X,y)
    #predictions = logmodel.predict(X_test)
    #predictions = logmodel.predict(X)
    return logmodel
    


# In[ ]:





# In[ ]:




