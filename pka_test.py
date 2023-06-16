# pip installs
#!pip install rdkit
#!pip install deepchem

import os, math
import pandas as pd
import numpy as np

import rdkit
import deepchem as dc

# plotting
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, mean_absolute_error
import lightgbm as lgb
from lightgbm import Dataset

from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from scipy import stats

from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("pkaclean.csv")

rdkit_featurizer = dc.feat.RDKitDescriptors(use_fragment=False, ipc_avg=False)
features = rdkit_featurizer(df.SMILES) # with one molecule

features.shape

column_names = rdkit_featurizer.descriptors

df0 = pd.DataFrame(data=features)
df0.columns = column_names

# adding molecule smiles and pka columns
df0["SMILES"] = df.SMILES
df0["pka_value"] = df.pka_value

df0.columns

df0.head()

y = df0.pka_value
X = df0.drop(columns=["SMILES", "pka_value"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

train_data = Dataset(X_train, label=y_train)
test_data = Dataset(X_test, label=y_test)

params = {
            "objective": "regression",
            "is_unbalance": "true",
            "boosting_type": "dart",
            "bagging_ratio": 0.6,
            "feature_fraction": 0.6,
            "metric": ["mse"],
        }

model = lgb.train(params=params,
                 train_set=train_data,
                 valid_sets=[test_data, train_data],
                  num_boost_round=14400,
                  verbose_eval=100
                 )

y_proba = model.predict(X_test)

tmp = pd.DataFrame(data={"y_true":  y_test, "y_proba": y_proba})
sns.scatterplot(x="y_true", y="y_proba", data=tmp);
plt.title("Real vs Predicted pKa")
plt.xlabel("Real pKa")
plt.ylabel("Predicted pKa")

mean_squared_error(y_test, y_proba), mean_absolute_error(y_test, y_proba)

stats.spearmanr(y_test, y_proba)

stats.pearsonr(y_test, y_proba)

r2_score(y_test, y_proba), mean_squared_error(y_test, y_proba)