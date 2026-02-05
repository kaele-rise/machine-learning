import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

scaler = StandardScaler()
x_train = np.array(train_data.drop(['id', 'SMILES', 'Tm'], axis=1))
scaler.fit(x_train)
x_train = scaler.transform(x_train)

y_train = np.array(train_data['Tm']).T

x_test = np.array(test_data.drop(['id', 'SMILES'], axis=1))
x_test = scaler.transform(x_test)


T = 30
max_depth = 3
algs = []
s = np.array(y_train.ravel())

for t in range(T):
    algs.append(DecisionTreeRegressor(max_depth=max_depth))
    algs[t].fit(x_train, s)
    s -= algs[t].predict(x_train)


y_test = algs[0].predict(x_test)
for i in range(1, T):
    y_test += algs[i].predict(x_test)

Qt = np.mean(s ** 2)

prediction = pd.DataFrame({'id': np.array(test_data['id']),
                           'Tm': y_test})

print(prediction.head())

prediction.to_csv('prediction.csv', index=False, encoding='utf-8')




