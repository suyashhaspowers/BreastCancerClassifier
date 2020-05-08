# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import sklearn


# %%
from sklearn.datasets import load_breast_cancer


# %%
from sklearn.model_selection import train_test_split


# %%
from sklearn.naive_bayes import GaussianNB


# %%
from sklearn.metrics import accuracy_score


# %%
import pandas as pd


# %%
import pickle


# %%
data = load_breast_cancer()


# %%
df = pd.DataFrame(data.data, columns=data.feature_names)
df = df[[col for col in df.columns if 'mean' in col]]


# %%
label_names = data['target_names'] # What we are trying to predict
labels = (data['target']) # Actual label data

# 0 - Malignant, 1 - Benign

feature_names = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension']
features = df


# %%
# Split our data, 2/3 - Train, 1/3 - Test
train, test, train_labels, test_labels = train_test_split(features,
                                                          labels,
                                                          test_size=0.33,
                                                          random_state=42)


# %%
# Initialize classifier
gnb = GaussianNB()

# Train classifier
model = gnb.fit(train, train_labels)


# %%
filename = 'finalized_model.sav'
pickle.dump(gnb, open(filename, 'wb'))


# %%
preds = gnb.predict(test)


# %%
print(preds)


# %%
# Evaluate accuracy
print(accuracy_score(test_labels, preds))

