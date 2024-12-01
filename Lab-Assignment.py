#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib


# In[2]:


dataset = pd.read_csv('Esophageal_Dataset.csv')


# In[4]:


threshold = len(dataset) * 0.5
dataset = dataset.dropna(thresh=threshold, axis=1)


# In[5]:


selected_columns = [
    "days_to_birth", "height", "weight", "vital_status",
    "tobacco_smoking_history", "frequency_of_alcohol_consumption",
    "primary_pathology_age_at_initial_pathologic_diagnosis", "person_neoplasm_cancer_status"
]


# In[6]:


dataset = dataset[selected_columns]


# In[7]:


dataset['vital_status'] = LabelEncoder().fit_transform(dataset['vital_status'])


# In[9]:


dataset.fillna(dataset.median(numeric_only=True), inplace=True)
for col in dataset.select_dtypes(include='object').columns:
    dataset[col].fillna(dataset[col].mode()[0], inplace=True)


# In[10]:


dataset = pd.get_dummies(dataset, drop_first=True)


# In[11]:


X = dataset.drop("vital_status", axis=1)
y = dataset["vital_status"]


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[15]:


model = LogisticRegression()
model.fit(X_train, y_train)


# In[16]:


joblib.dump(model, "heart_disease_model.pkl")


# In[17]:


y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# In[ ]:




