# install libraries

# !pip install rdkit
# !pip install deepchem

import os, math
import pandas as pd
import numpy as np

import rdkit
import deepchem as dc

# plotting
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("delaney-processed.csv")
# df = pd.read_csv("curated-solubility-dataset.csv")

rdkit_featurizer = dc.feat.RDKitDescriptors(use_fragment=False, ipc_avg=False)
features = rdkit_featurizer(df.SMILES) # with one molecule

features.shape

# Creating the feature dataset

column_names = rdkit_featurizer.descriptors

df0 = pd.DataFrame(data=features)
df0.columns = column_names

# adding molecule ids and solubility columns
# 'ESOL' for delaney and 'Solubility' for Aqsoldb
df0["SMILES"] = df.SMILES
df0["ESOL"] = df.ESOL

df0.columns
df0.head()

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, mean_absolute_error
import lightgbm as lgb
from lightgbm import Dataset

y = df0.ESOL
X = df0.drop(columns=["SMILES", "ESOL"])


#Training and validation

import math
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from scipy import stats

# Split the data into training and combined validation/testing set
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Split the combined validation/testing set into validation and testing set
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# Define the LightGBM parameters
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting': 'gbdt',
    'num_leaves': 40,
    'learning_rate': 0.1,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

# Convert the data to LightGBM Dataset format
data = lgb.Dataset(X_train_val, label=y_train_val)

# Perform cross-validation
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
mse_scores = []
mae_scores = []

for train_index, val_index in kf.split(X_train_val):
    X_train, X_val = X_train_val.iloc[train_index], X_train_val.iloc[val_index]
    y_train, y_val = y_train_val.iloc[train_index], y_train_val.iloc[val_index]

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)

    model = lgb.train(params, train_data, num_boost_round=1000, valid_sets=[train_data, val_data], early_stopping_rounds=10)

    y_val_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_val_pred)
    mae = mean_absolute_error(y_val, y_val_pred)

    mse_scores.append(mse)
    mae_scores.append(mae)

avg_mse = np.mean(mse_scores)
avg_mae = np.mean(mae_scores)

print("Average Mean Squared Error:", avg_mse)
print("Average Mean Absolute Error:", avg_mae)

# Train final model on the entire training dataset
final_model = lgb.train(params, data, num_boost_round=1000)

# Make predictions on the testing set
y_test_pred = final_model.predict(X_test)

# Evaluate the final model on the testing set
mse_test = mean_squared_error(y_test, y_test_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
r = r2_score(y_test, y_test_pred)

performance = pd.DataFrame([[mse_test], [mae_test], [r], [math.sqrt(mse_test)]],
                   columns=['Final Model Test Set'], index = ['Mean Square Error', 'Mean Absolute Error', 'R^2', 'RMSE'])
performance

# Spearman correlation
stats.spearmanr(y_test, y_test_pred)

# Pearson correlation
stats.pearsonr(y_test, y_test_pred)

tmp = pd.DataFrame(data={"y_true":  y_test, "y_proba": y_test_pred})
sns.scatterplot(x="y_true", y="y_proba", data=tmp);
plt.title("Real vs Predicted Solubility")
plt.xlabel("Real Solubility")
plt.ylabel("Predicted Solubility");
