import pandas as pd
import pickle
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import yaml
import json
import os
from dvclive import Live


test = pd.read_csv('./data/processed/test_processed.csv')

X_test = test.drop(columns=['Placed'])
y_test = test['Placed']

rf = pickle.load(open('model.pkl','rb'))

y_pred = rf.predict(X_test)


accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)
f1_score = f1_score(y_test,y_pred)

with open('params.yaml','r') as file:
    params = yaml.safe_load(file)


with Live(save_dvc_exp=True) as live:
    live.log_metric('accuracy',accuracy)
    live.log_metric('precision',precision)
    live.log_metric('recall',recall)
    live.log_metric('f1_score',f1_score)

    for param,value in params.items():
        for key,val in value.items():
            live.log_param(f'{param}_{key}',val)

metrics = {
    'accuracy':accuracy,
    'precision':precision,
    'recall':recall,
    'f1_score':f1_score
}


with open('metrics.json','w') as f:
    json.dump(metrics,f,indent=4)

