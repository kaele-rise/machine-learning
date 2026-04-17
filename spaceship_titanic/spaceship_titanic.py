import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
import numpy as np

# импорт данных
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

X_train = train_data[['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Age',
       'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']]
y_train = train_data['Transported']

X_test = test_data[['PassengerId', 'HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Age',
       'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']]

num_columns = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
cat_columns = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']

# пайплайн
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_columns),
    ('cat', cat_pipeline, cat_columns)
])

pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('model', LogisticRegression())
])


pipeline.fit(X_train, y_train)

predict = pipeline.predict(X_test)

result = pd.DataFrame(
       {'PassengerId': X_test['PassengerId'],
        'Transported': predict
        })

result.to_csv('prediction.csv', index=False)



