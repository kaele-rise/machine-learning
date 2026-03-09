import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

x_train = train_data.drop(['id', 'Churn'], axis=1)
y_train = train_data['Churn'].replace({'Yes': 1, 'No': 0})
y_train = y_train.astype('int')

x_test = test_data.drop(['id'], axis=1)

num_columns = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
cat_columns = ['gender', 'Partner', 'Dependents',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']


cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore')),
])

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

preprocessor = ColumnTransformer([
    ('cat, cat', cat_pipeline, cat_columns),
    ('num', num_pipeline, num_columns)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('voting_model', VotingClassifier(estimators=[('log_regr', LogisticRegression(class_weight='balanced', max_iter=500)),
                                                  ('d_tree', DecisionTreeClassifier(max_depth=4, class_weight='balanced')),
                                                  ('svm', SVC(kernel='linear', probability=True, random_state=42)),
                                                  ('knn', KNeighborsClassifier(n_neighbors=5))], n_jobs=-1))
])

pipeline.fit(x_train, y_train)

y_pred = pipeline.predict(x_test)
predict = pd.DataFrame({'id': [i for i in range(len(y_pred))],
                        'Churn': y_pred})
predict.to_csv('prediction.csv', index=False, encoding='utf-8')


