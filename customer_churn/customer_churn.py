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
from tqdm import tqdm

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

x_train_processed = preprocessor.fit_transform(x_train)
x_test_processed = preprocessor.transform(x_test)

estimators = [
    ('log_regr', LogisticRegression(class_weight='balanced', max_iter=500)),
    ('d_tree', DecisionTreeClassifier(max_depth=4, class_weight='balanced')),
    ('svm', SVC(kernel='linear', random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5))
]

trained_estimators = []
for name, model in tqdm(estimators, desc="Training"):
    model.fit(x_train_processed, y_train)
    trained_estimators.append((name, model))

voting_model = VotingClassifier(estimators=trained_estimators, n_jobs=-1)

y_pred = voting_model.predict(x_test_processed)
predict = pd.DataFrame({'id': [i for i in range(len(y_pred))],
                        'Churn': y_pred})
predict.to_csv('prediction.csv', index=False, encoding='utf-8')


