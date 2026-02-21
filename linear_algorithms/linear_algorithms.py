import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, Ridge, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


test_data = pd.read_csv('test.csv')
train_data = pd.read_csv('train.csv')

cat_columns = ['Sex', 'HasPartner', 'HasChild', 'HasPhoneService',
       'HasMultiplePhoneNumbers', 'HasInternetService',
       'HasOnlineSecurityService', 'HasOnlineBackup', 'HasDeviceProtection',
       'HasTechSupportAccess', 'HasOnlineTV', 'HasMovieSubscription',
       'HasContractPhone', 'IsBillingPaperless', 'PaymentMethod']

num_columns = ['ClientPeriod', 'MonthlySpending', 'TotalSpent', 'IsSeniorCitizen']

for col in num_columns:
    train_data[col] = pd.to_numeric(train_data[col], errors='coerce')
    test_data[col] = pd.to_numeric(test_data[col], errors='coerce')

x_train = train_data.drop('Churn', axis=1)
y_train = train_data['Churn']
x_test = test_data



# preprocessing pipeline
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer([
    ('cat', cat_pipeline, cat_columns),
    ('num', num_pipeline, num_columns)
])


models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
}

x_train_, x_test_, y_train_, y_test_ = train_test_split(x_train, y_train, test_size=0.2, random_state=42)


predictions = {}
metrics = {}

for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    pipeline.fit(x_train_, y_train_)

    y_pred_ = pipeline.predict(x_test_)
    predictions[name] = y_pred_

    mse = mean_squared_error(y_test_, y_pred_)
    r2 = r2_score(y_test_, y_pred_)
    metrics[name] = {'MSE': mse, 'R2': r2}


print(metrics)


