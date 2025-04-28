#!/usr/bin/env python
# coding: utf-8

# In[42]:


# Exploratory Data Analysis (EDA) on train.csv

# Step 1: Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Display plots inline
get_ipython().run_line_magic('matplotlib', 'inline')

# Step 2: Load Data
# Load dataset
data = pd.read_csv("C:/Users/nikhi/OneDrive/Desktop/train.csv")


# In[14]:


Step 3
# Checking first few rows
data.head()


# In[9]:


# Basic info
data.info()


# In[11]:


# Summary statistics
data.describe()


# In[13]:


# Checking for missing values
data.isnull().sum()


# In[16]:


# Checking for duplicate entries
data.duplicated().sum()


# In[19]:


# Step 4: Univariate Analysis
# Identify categorical and numerical columns
categorical_cols = data.select_dtypes(include=['object']).columns
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns


# In[44]:


# Plot categorical columns
for col in categorical_cols:
    plt.figure(figsize=(6,4))
    data[col].value_counts().plot(kind='bar')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()


# In[34]:


# Plot numerical columns
for col in numerical_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(data[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.show()

    plt.figure(figsize=(6,4))
    sns.boxplot(x=data[col])
    plt.title(f'Boxplot of {col}')
    plt.show()


# In[37]:


# Step 5: Bivariate/Multivariate Analysis
# Correlation Matrix
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Pairplot (limited to avoid heavy plots)
sns.pairplot(data[numerical_cols[:4]])  # Plot only first 4 numeric features
plt.show()


# In[38]:


if len(numerical_cols) >= 2:
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=data[numerical_cols[0]], y=data[numerical_cols[1]])
    plt.title(f'Scatterplot: {numerical_cols[0]} vs {numerical_cols[1]}')
    plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




