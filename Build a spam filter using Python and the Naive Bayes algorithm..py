#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 📦 Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer  # Converts text into TF-IDF features
from sklearn.model_selection import train_test_split         # For splitting dataset
from sklearn.naive_bayes import MultinomialNB                # 👉 Naive Bayes classifier for text data
from sklearn.metrics import accuracy_score, classification_report
import chardet  # Detects file encoding

# ----------------------------------------
# 📁 Step 1: Detect encoding of CSV file to avoid reading errors
file_path = r"C:\Users\91989\Downloads\archive (3)\spam.csv"  # Replace with your file path
with open(file_path, "rb") as f:
    result = chardet.detect(f.read(100000))  # Read first 100,000 bytes for encoding detection
    detected_encoding = result['encoding']

# ----------------------------------------
# 📄 Step 2: Load dataset using the detected encoding
df = pd.read_csv(file_path, encoding=detected_encoding)

# ----------------------------------------
# 🧹 Step 3: Keep only necessary columns and rename them
df = df[['v1', 'v2']]  # Keep 'label' and 'message' columns
df.columns = ['label', 'message']  # Rename for clarity

# ----------------------------------------
# 🔁 Step 4: Convert 'ham' to 0 and 'spam' to 1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# ----------------------------------------
# ✂️ Step 5: Split data into training (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# ----------------------------------------
# 🔢 Step 6: Convert messages to TF-IDF vectors
# TF-IDF: Term Frequency-Inverse Document Frequency to highlight important words
vectorizer = TfidfVectorizer(stop_words='english')  # Remove common stopwords
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ----------------------------------------
# 🧠 Step 7: Train the Naive Bayes model
# 👉 Here, Naive Bayes is learning word probabilities for spam and ham
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)

# ----------------------------------------
# 🔎 Step 8: Predict labels for test data
# 👉 This is where the spam filter is used to classify messages
y_pred = nb.predict(X_test_tfidf)

# ----------------------------------------
# 📊 Step 9: Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# ----------------------------------------
# 🧪 Step 10: Test the model on new, unseen messages
test_messages = [
    "Congratulations You have won a $1000 gift card Claim now",
    "Hey are we still meeting for lunch today",
    "Urgent Your account has been compromised Click the link to reset your password"
]

# Convert new messages to TF-IDF format
test_tfidf = vectorizer.transform(test_messages)

# Predict whether each message is spam or ham
predictions = nb.predict(test_tfidf)

# ----------------------------------------
# 📤 Step 11: Output predictions
# 👉 The spam filter is applied here to real-world examples
for msg, pred in zip(test_messages, predictions):
    print("\nMessage:", msg)
    print("Prediction:", "Spam" if pred == 1 else "Ham")


# In[ ]:




