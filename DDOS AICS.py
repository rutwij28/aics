#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install --upgrade pip setuptools wheel')
get_ipython().system('pip uninstall -y numpy pandas')
get_ipython().system('pip install numpy==1.24.4 pandas==1.5.3')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Correct the file path (make sure it's valid and does not contain hidden characters)
data = pd.read_csv(r"C:\Users\91989\Downloads\DDos.csv")

print('Missing values before handling:\n', data.isnull().sum())

# Handle missing and infinite values
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

print('Missing values after handling:\n', data.isnull().sum())

# Label encode categorical columns
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Check for remaining NaNs or infinite values
if X_train.isnull().sum().sum() > 0 or np.any(np.isinf(X_train)):
    print("There are still NaN or infinity values in the training data.")

# Final cleaning
X_train = X_train.dropna()
X_train = X_train[~np.isinf(X_train).any(axis=1)]

# Train Random Forest model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
print("Training the model...")
rf_classifier.fit(X_train, y_train)
print("Model trained successfully.")

# Predictions and evaluation
print("Making predictions...")
y_pred = rf_classifier.predict(X_test)

print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
print('\nClassification Report:')
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'DDoS'], yticklabels=['Normal', 'DDoS'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()


# In[ ]:




