import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score

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
    'Linear Regression': LogisticRegression(),
    'Ridge': Perceptron(),
    'Lasso': RidgeClassifier(),
    'ElasticNet': LinearDiscriminantAnalysis(),
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

    model_metrics = {}
    if hasattr(model, 'predict_proba'):
        auc = roc_auc_score(y_test_, pipeline.predict_proba(x_test_)[:, 1])
    else:
        calibrated = CalibratedClassifierCV(pipeline, method='sigmoid', cv=5)
        calibrated.fit(x_train_, y_train_)
        prob = calibrated.predict_proba(x_test_)
        auc = roc_auc_score(y_test_, prob[:, 1])

    model_metrics['AUC-ROC'] = auc
    f1 = f1_score(y_test_, y_pred_, average='macro')
    model_metrics['F1'] = f1
    metrics[name] = model_metrics


print(metrics)


prediction_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LogisticRegression())
])
prediction_pipeline.fit(x_train, y_train)
y_pred = prediction_pipeline.predict(x_test)

prediction = pd.DataFrame({
    'Id': [i for i in range(len(y_pred))],
    'Churn': y_pred
})
prediction.to_csv('prediction.csv', index=False, encoding='utf-8')





