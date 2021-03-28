# hit_predictor
# Libraries

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import warnings
warnings.filterwarnings(action='ignore')

# Dataset
dfs = [pd.read_csv(f'../input/the-spotify-hit-predictor-dataset/dataset-of-{decade}0s.csv') for decade in['6', '7', '8', '9', '0', '1']]
dfs[0]

# Shuffle the data
for i, decade in enumerate([1960, 1970, 1980, 1990, 2000, 2010]):
    dfs[i]['decade'] = pd.Series(decade, index=dfs[i].index)
    
data = pd.concat(dfs, axis=0).sample(frac=1.0, random_state=1).reset_index(drop=True)
data
data.info()

# Preprocessing
def preprocess_inputs(df):
    df = df.copy()
    
    # Drop high-cardinality categorical columns
    df = df.drop(['track', 'artist', 'uri'], axis=1)
    
    # Split df into x and y
    y = df['target']
    x = df.drop('target', axis=1)
    
    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=1)
   
    # Scale x
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = pd.DataFrame(scaler.transform(x_train), index=x_train.index, columns=x_train.columns)
    x_test = pd.DataFrame(scaler.transform(x_test), index=x_test.index, columns=x_test.columns)

    return x_train, x_test, y_train, y_test 
    
x_train, x_test, y_train, y_test = preprocess_inputs(data)
x_train
y_train

# Training
models = {
    "                   Logistic Regression": LogisticRegression(),
    "                   K-Nearest Neighbors": KNeighborsClassifier(),
    "                         Decision Tree": DecisionTreeClassifier(),
    "Support Vector Machine (Linear Kernel)": LinearSVC(),
    "   Support Vector Machine (RBF Kernel)": SVC(),
    "                        Neural Network": MLPClassifier(),
    "                         Random Forest": RandomForestClassifier(),
    "                     Gradient Boosting": GradientBoostingClassifier()
    
}

for name, model in models.items():
    model.fit(x_train, y_train)
    print(name + " trained.")
    
# Results
for name, model in models.items():
    print(name + ": {:.2f}%".format(model.score(x_test,y_test)* 100))
