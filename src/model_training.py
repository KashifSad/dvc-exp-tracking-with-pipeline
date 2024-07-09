import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
import os
import yaml


with open('params.yaml','r') as file:
    params = yaml.safe_load(file)


n_estimators = params['model_training']['n_estimators']
max_depth = params['model_training']['max_depth']
bootstrap = params['model_training']['bootstrap']
criterion = params['model_training']['criterion']


train = pd.read_csv('./data/processed/train_processed.csv')

X_train = train.drop(columns=["Placed"])
y_train = train["Placed"]

rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,bootstrap=bootstrap,criterion=criterion)
rf.fit(X_train,y_train)

pickle.dump(rf,open('model.pkl','wb'))



