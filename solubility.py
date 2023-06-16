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

# Use delaney dataset in this case
df = pd.read_csv("delaney-processed.csv")

rdkit_featurizer = dc.feat.RDKitDescriptors(use_fragment=False, ipc_avg=False)
features = rdkit_featurizer(df.SMILES) # with one molecule

features.shape

# Creating the feature dataset

column_names = rdkit_featurizer.descriptors

df0 = pd.DataFrame(data=features)
df0.columns = column_names

# adding molecule smiles and solubility columns
df0["SMILES"] = df.SMILES
df0["ESOL"] = df.ESOL

# Initially thought classification, but regression was better
# not needed at the moment
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score


def optimal_cutoff(target, predicted):
    """ determine optimal probability cutoff point for classification
    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return round(list(roc_t['threshold'])[0], 2)

def make_confusion_matrix(y_true, y_pred):
    '''A confusion matrix plotter'''

    conf_matrix = confusion_matrix(y_true, y_pred)
    data = conf_matrix.transpose()

    _, ax = plt.subplots()
    ax.matshow(data, cmap="Blues")

    for (i, j), z in np.ndenumerate(data):
        ax.text(j, i, '{}'.format(z), ha='center', va='center')

    plt.xticks([])
    plt.yticks([])
    plt.title("T labels\n 0  {}     1\n".format(" "*18), fontsize=11)
    plt.ylabel("P labels\n 1   {}     0".format(" "*18), fontsize=11)

def roc(y_true, y_proba):
    ''' ROC curve with appropriate labels and legend '''
    fpr, tpr, _ = roc_curve(y_true, y_proba)

    _, ax = plt.subplots()

    ax.plot(fpr, tpr, color='r');
    ax.plot([0, 1], [0, 1], color='y', linestyle='--')
    ax.fill_between(fpr, tpr, label=f"AUC: {round(roc_auc_score(y_true, y_proba), 3)}")
    ax.set_aspect(0.90)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlim(-0.02, 1.02);
    ax.set_ylim(-0.02, 1.02);
    plt.legend()
    plt.show()


def summarize_results(y_true, y_pred):
    '''
    Use real lebels and the predict label probabilties to print some metrics out
    '''
    print(accuracy_score(y_true, y_pred).round(2))
    conf_matrix = confusion_matrix(y_true, y_pred)
    sensitivity = round(conf_matrix[1, 1]/(conf_matrix[1, 1] + conf_matrix[1, 0]), 2)
    specificity = round(conf_matrix[0, 0]/(conf_matrix[0, 0] + conf_matrix[0, 1]), 2)

    ppv = round(conf_matrix[1, 1]/(conf_matrix[1, 1] + conf_matrix[0, 1]), 2)
    npv = round(conf_matrix[0, 0]/(conf_matrix[0, 0] + conf_matrix[1, 0]), 2)

    print(sensitivity)
    print(specificity)

    print(ppv)
    print(npv)

    print(precision_score(y_true, y_pred).round(2))
    print(recall_score(y_true, y_pred).round(2))

df0.columns

df0.head()

y = df0.ESOL
X = df0.drop(columns=["SMILES", "ESOL"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

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
                  num_boost_round=7200,
                  verbose_eval=100
                 )

y_proba = model.predict(X_test)

tmp = pd.DataFrame(data={"y_true":  y_test, "y_proba": y_proba})
sns.scatterplot(x="y_true", y="y_proba", data=tmp);
plt.title("Real vs Predicted Solubility")
plt.xlabel("Real Solubility")
plt.ylabel("Predicted Solubility")

mean_squared_error(y_test, y_proba), mean_absolute_error(y_test, y_proba)
r2_score(y_test, y_proba), mean_squared_error(y_test, y_proba)


