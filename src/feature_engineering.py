import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import yaml

train = pd.read_csv('./data/raw/train.csv')
test = pd.read_csv('./data/raw/test.csv')


with open('params.yaml','r') as file:
    params = yaml.safe_load(file)


n_components = params['feature_engineering']['n_components']


X_train = train.drop(columns=['Placed'])
y_train = train['Placed']
X_test = test.drop(columns=['Placed'])
y_test = test['Placed']


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)


pca  = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_test_scaled)
X_test_pca = pca.fit_transform(X_test_scaled)


train_processed = pd.DataFrame(X_train_pca, columns=[f'PC{i+1}'for i in range(3)])
train_processed['Placed'] = y_train.reset_index(drop=True)
test_processed = pd.DataFrame(X_test_pca,columns=[f'PC{i+1}'for i in range(3)])
test_processed['Placed'] = y_test.reset_index(drop=True)


train_processed.to_csv('./data/processed/train_processed.csv',index=False)
test_processed.to_csv('./data/processed/test_processed.csv',index=False)