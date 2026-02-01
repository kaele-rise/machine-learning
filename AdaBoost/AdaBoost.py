import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

feature_cols = ['day', 'pressure', 'maxtemp', 'temparature', 'mintemp',
               'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection',
               'windspeed']

x_train = train_data[feature_cols].fillna(train_data[feature_cols].mean())
x_test = test_data[feature_cols].fillna(train_data[feature_cols].mean())

y_train = train_data['rainfall']
x_train, y_train, x_test = np.array(x_train), np.array(y_train), np.array(x_test)

y_train[y_train == 0] = -1


max_depth = 3
w = np.ones(len(x_train)) / len(x_train) #
T = 10

algs = []
alfa = []

for t in range(T):
    algs.append(DecisionTreeClassifier(criterion='gini', max_depth=max_depth))
    algs[t].fit(x_train, y_train)
    predict = algs[t].predict(x_train)
    N = np.sum((y_train != predict) * w) + 1e-8

    alfa.append(0.5 * np.log((1 - N) / N))
    w *= np.exp(-1 * alfa[t] * y_train * predict)
    w /= np.sum(w)


y_test = alfa[0] * algs[0].predict(x_test)
for n in range(1, T):
    y_test += alfa[n] * algs[n].predict(x_test)

y_test = np.sign(y_test)
test_len = len(y_test)
y_test[y_test == -1] = 0
submission = pd.DataFrame({'id': [i for i in range(test_len)],
                       'rainfall': y_test})

submission.to_csv('data.csv', index=False, encoding='utf-8')

