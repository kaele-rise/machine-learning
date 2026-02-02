import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

x_train = train_data.drop(['id', 'SMILES', 'Tm'], axis=1)
y_train = train_data['Tm']

x_test = test_data.drop(['id', 'SMILES'], axis=1)


