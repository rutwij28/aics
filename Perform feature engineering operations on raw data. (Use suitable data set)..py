#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv(r"C:\Users\91989\Downloads\Titanic-Dataset - Titanic-Dataset.csv")

# Clean column names (remove leading/trailing spaces)
df.columns = df.columns.str.strip()

# Display column names for debugging
print("Columns in the dataset:", df.columns.tolist())

# Fill missing values
if 'Age' in df.columns:
    df['Age'].fillna(df['Age'].median(), inplace=True)

if 'Embarked' in df.columns:
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Feature engineering
if 'SibSp' in df.columns and 'Parch' in df.columns:
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# One-hot encoding safely
cols_to_encode = [col for col in ['Embarked', 'Sex'] if col in df.columns]
df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)

# Drop irrelevant columns if present
drop_cols = ['Name', 'Ticket', 'Cabin', 'PassengerId', 'SibSp', 'Parch']
df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

# Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Unit test to verify expected columns
def test_preprocessed_data_columns():
    expected_columns = ['Survived', 'Pclass', 'Age', 'Fare', 'FamilySize', 'Embarked_Q', 'Embarked_S', 'Sex_male']
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        raise AssertionError(f"Missing expected columns: {missing}")
    return "All tests passed."

# Run test
print(test_preprocessed_data_columns())


# In[ ]:




