from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

x_train = train_data.drop(['id', 'Heart Disease'], axis=1)
y_train = train_data['Heart Disease']
y_train = y_train.map({
    'Presence': 1,
    'Absence': 0
})

x_test = test_data.drop(['id'], axis=1)

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('model', GradientBoostingClassifier())
])

pipeline.fit(x_train, y_train)

y_test = pipeline.predict(x_test)

prediction = pd.DataFrame({
    'id': np.array(test_data['id']),
    'Heart Disease': y_test
})

prediction.to_csv('prediction.csv', index=False, encoding='utf-8')