import numpy as np
import pandas as pd
import requests as re
import zipfile
import io
path = "mydata/"
data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00341/HAPT%20Data%20Set.zip'
data = re.get(data_url)
zip_data = zipfile.ZipFile(io.BytesIO(data.content))
zip_data.extractall(path=path)
features_path = f"{path}features.txt"
x_train_path = f"{path}Train/X_train.txt"
y_train_path = f"{path}Train/y_train.txt"
x_test_path = f"{path}Test/X_test.txt"
y_test_path = f"{path}Test/y_test.txt"
with open(features_path, 'r') as f:
    columns = ["".join(elt.split()) for elt in f.readlines()]
ytrain = pd.read_csv(y_train_path, sep =" ", header=None)
xtrain = pd.read_csv(x_train_path, sep =" ", header=None)
xtrain.columns = columns
ytest = pd.read_csv(y_test_path, sep =" ", header=None)
xtest = pd.read_csv(x_test_path, sep =" ", header=None)
xtest.columns = columns
x = pd.concat([xtrain, xtest]).reset_index(drop=True)
y = pd.concat([ytrain, ytest]).reset_index(drop=True)
x["target"] = y
import matplotlib.pyplot as plt
import seaborn as sns
corr_matrix = x.corr()
best_10_features = corr_matrix["target"].apply(lambda x: np.abs(x)).sort_values(ascending=False).iloc[1:20]
cols = ["target"] + best_10_features.index.tolist()
corr_bests = corr_matrix.loc[cols, cols]
mask = np.triu(np.ones_like(corr_bests, dtype=np.bool))
f, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(corr_bests, mask=mask)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
result = []
for estimator in [DecisionTreeClassifier(),
                  LogisticRegression(),
                  RandomForestClassifier(),
                  AdaBoostClassifier()]:
    estimator.fit(xtrain, ytrain)
    prediction = estimator.predict(xtest)
    result.append({"name":estimator.__class__.__name__,
                   "precision":precision_score(ytest, prediction, average="weighted"),
                   "recall":recall_score(ytest, prediction, average="weighted"),
                   "f1_score": f1_score(ytest, prediction, average="weighted")})
f, ax = plt.subplots(figsize=(10, 7))
pd.DataFrame(result).set_index("name").plot(kind="bar")
logreg = LogisticRegression()
logreg.get_params()
grid={"C":np.logspace(-2,2,4), "penalty":["l1","l2"], "tol":np.logspace(-5,-2,4)}
from sklearn.model_selection import GridSearchCV
logreg = LogisticRegression(multi_class="auto", solver="saga")
logregcv = GridSearchCV(logreg,
                        grid, cv=10,
                        verbose=1,
                        n_jobs=-1)

logregcv.fit(xtrain, np.ravel(ytrain),)
precision_score(ytest, logregcv.predict(xtest), average="weighted")
from flask import Flask
path = "data/"
app = Flask(__name__)
features_path = f"{path}features.txt"
x_train_path = f"{path}Train/X_train.txt"
y_train_path = f"{path}Train/y_train.txt"
x_test_path = f"{path}Test/X_test.txt"
y_test_path = f"{path}Test/y_test.txt"
data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00341/HAPT%20Data%20Set.zip'
@app.route('/')
@app.route('/health')
def health():
    return {"success":True, "message":"Api is running fine"}
@app.route('/download')
def download():
    data = re.get(data_url)
    zip_data = zipfile.ZipFile(io.BytesIO(data.content))
    zip_data.extractall(path=path)
    return {"success":True, "message":f"Successfully download data at {path}"}
                   
@app.route('/train')
def train():
    y_train = pd.read_csv(y_train_path, sep =" ", header=None)
    x_train = pd.read_csv(x_train_path, sep =" ", header=None)
    x_train.columns = columns

    y_test = pd.read_csv(y_test_path, sep =" ", header=None)
    x_test = pd.read_csv(x_test_path, sep =" ", header=None)
    x_test.columns = columns
    for estimator in [DecisionTreeClassifier(),LogisticRegression(),RandomForestClassifier(),AdaBoostClassifier()]:
        estimator.fit(x_train, y_train)
        prediction = estimator.predict(x_test)
    
        result.append({"name":estimator.__class__.__name__,
                       "precision":precision_score(y_test, prediction, average="weighted"),
                       "recall":recall_score(y_test, prediction, average="weighted"),
                       "f1_score": f1_score(y_test, prediction, average="weighted")})
    return {"success":True, "message":"Successfully trained model", "data":{"results":result}}
    
import request
@app.route('/predict', methods=["POST"])
def predict():
    if request.method == 'POST':
        data = request.get_json()
        if "data" not in data:
            return {"success": False,
                    "message": "You must provide json with data key"}

        prediction, prediction_proba = estimator.predict(data["data"]), estimator.predict_proba(data["data"])
        return {"success": True, "result": {
            "classes": [int(elt) for elt in prediction],
            "probabilities": [[float(elt) for elt in probas] for probas in prediction_proba],
        }}
app.run()




