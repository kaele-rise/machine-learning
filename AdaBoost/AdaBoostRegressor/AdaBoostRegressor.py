import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler


class AdaBoostRegressor:
    def __init__(self, max_depth=3, iter=30):
        self.max_depth = max_depth
        self.iter = iter
        self.algs = []

    def fit(self, x, y):
        s = np.array(y.ravel())

        for t in range(self.iter):
            self.algs.append(DecisionTreeRegressor(max_depth=self.max_depth))
            self.algs[t].fit(x, s)
            s -= self.algs[t].predict(x)

        return self

    def predict(self, x):
        self.y_test = self.algs[0].predict(x)
        for i in range(1, self.iter):
            self.y_test += self.algs[i].predict(x)

        return self.y_test


pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('scaler', StandardScaler()),
    ('model', AdaBoostRegressor())
])

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

x_train = np.array(train_data.drop(['id', 'SMILES', 'Tm'], axis=1))
y_train = np.array(train_data['Tm']).T
x_test = np.array(test_data.drop(['id', 'SMILES'], axis=1))


pipeline.fit(x_train, y_train)
y_test = pipeline.predict(x_test)

prediction = pd.DataFrame({'id': np.array(test_data['id']),
                           'Tm': y_test})

print(prediction.head())

prediction.to_csv('prediction.csv', index=False, encoding='utf-8')




