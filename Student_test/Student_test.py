import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

x_train = train_data[['age', 'gender', 'course', 'study_hours', 'class_attendance',
       'internet_access', 'sleep_hours', 'sleep_quality', 'study_method',
       'facility_rating', 'exam_difficulty']]

y_train = train_data['exam_score']

x_test = test_data[['age', 'gender', 'course', 'study_hours', 'class_attendance',
       'internet_access', 'sleep_hours', 'sleep_quality', 'study_method',
       'facility_rating', 'exam_difficulty']]

num_columns = ['age', 'study_hours', 'class_attendance', 'sleep_hours']
cat_columns = ['gender', 'course', 'internet_access', 'sleep_quality', 'study_method', 'facility_rating', 'exam_difficulty']

num_pipeline = Pipeline([
       ('imputer', SimpleImputer(strategy='mean')),
       ('scaler', StandardScaler())
])
cat_pipeline = Pipeline([
       ('imputer', SimpleImputer(strategy='most_frequent')),
       ('encoder', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
       ('num', num_pipeline, num_columns),
       ('cat', cat_pipeline, cat_columns)
])
pipeline = Pipeline([
       ('preprocessor', preprocessor),
       ('model', LinearRegression())
])

pipeline.fit(x_train, y_train)

predict = pipeline.predict(x_test)
prediction_data = pd.DataFrame({
       'id': [i for i in range(630000, 630000 + len(predict))],
       'exam_score': predict
})
prediction_data.to_csv('predict.csv', encoding='utf-8', index=False)


