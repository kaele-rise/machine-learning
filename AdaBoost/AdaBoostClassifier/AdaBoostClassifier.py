import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class AdaBoostClassifier:
    def __init__(self, max_depth=3, iter=10):
        self.max_depth = max_depth
        self.iter = iter
        self.algs = []
        self.alpha = []
        self.w = None

    def fit(self, x, y):
        self.w = np.ones(len(x)) / len(x)  #

        for t in range(self.iter):
            self.algs.append(DecisionTreeClassifier(criterion='gini', max_depth=self.max_depth))
            self.algs[t].fit(x, y, sample_weight=self.w)
            self.prediction = self.algs[t].predict(x)
            self.N = np.sum(self.w * (y != self.prediction)) + 1e-8

            self.alpha.append(0.5 * np.log((1 - self.N) / self.N))
            self.w *= np.exp(-1 * self.alpha[t] * y * self.prediction)
            self.w /= np.sum(self.w)

        return self

    def predict(self, x):
        self.y_test = self.alpha[0] * self.algs[0].predict(x)
        for n in range(1, self.iter):
            self.y_test += self.alpha[n] * self.algs[n].predict(x)

        self.y_test = np.sign(self.y_test)
        self.y_test[self.y_test == -1] = 0
        return self.y_test

pipline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('model', AdaBoostClassifier())
])



train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

feature_cols = ['pressure', 'maxtemp', 'temparature', 'mintemp',
               'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection',
               'windspeed']


x_train = train_data[feature_cols]
x_test = test_data[feature_cols]
y_train = train_data['rainfall'].copy()
y_train[y_train == 0] = -1



pipline.fit(x_train, y_train)
y_test = pipline.predict(x_test)

test_len = len(y_test)
submission = pd.DataFrame({'id': [i for i in range(2190, 2190 + test_len)],
                       'rainfall': y_test})

submission.to_csv('data.csv', index=False, encoding='utf-8')

