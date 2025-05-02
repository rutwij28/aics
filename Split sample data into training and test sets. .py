#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Install necessary libraries if not already installed (run in a Jupyter cell)
# !pip install pandas scikit-learn

# -----------------------------
# 1. Import Libraries
# -----------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -----------------------------
# 2. Load the Dataset
# -----------------------------
iris_df = pd.read_csv(r"C:\Users\91989\Downloads\iris.csv")
print("----------------------------------------------------------------------------------")
print("Iris Dataset")
print("----------------------------------------------------------------------------------")
print(iris_df)
print("----------------------------------------------------------------------------------")

# -----------------------------
# 3. Explore the Dataset
# -----------------------------
print("First Few Rows in Iris Dataset")
print("----------------------------------------------------------------------------------")
print(iris_df.head())
print("----------------------------------------------------------------------------------")

print("Information about Iris Dataset")
print("----------------------------------------------------------------------------------")
print(iris_df.info())
print("----------------------------------------------------------------------------------")

print("Summary Statistics about Iris Dataset")
print("----------------------------------------------------------------------------------")
print(iris_df.describe())
print("----------------------------------------------------------------------------------")

# -----------------------------
# 4. Data Preprocessing
# -----------------------------
# Drop only 'species' as there is no 'Id' column
X = iris_df.drop(['species'], axis=1)
y = iris_df['species']

# Encode target labels as integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# -----------------------------
# 5. Train the Model
# -----------------------------
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# -----------------------------
# 6. Evaluate the Model
# -----------------------------
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy score: {accuracy:.2f}')
print("----------------------------------------------------------------------------------")

report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print("Classification Report:\n", report)
print("----------------------------------------------------------------------------------")

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)
print("----------------------------------------------------------------------------------")

# -----------------------------
# 7. Predict New Samples
# -----------------------------
new_data = [[5.1, 3.5, 1.4, 0.2]]
predicted_class = label_encoder.inverse_transform(clf.predict(new_data))
print(f"Input: {new_data} => Predicted class: {predicted_class[0]}")
print("----------------------------------------------------------------------------------")

new_data = [[8.1, 3.5, 5.4, 0.2]]
predicted_class = label_encoder.inverse_transform(clf.predict(new_data))
print(f"Input: {new_data} => Predicted class: {predicted_class[0]}")
print("----------------------------------------------------------------------------------")


# In[ ]:




