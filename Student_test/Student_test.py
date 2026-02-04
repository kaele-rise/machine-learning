import pandas as pd

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

x_train = train_data[['age', 'gender', 'course', 'study_hours', 'class_attendance',
       'internet_access', 'sleep_hours', 'sleep_quality', 'study_method',
       'facility_rating', 'exam_difficulty']]
x_train.loc[:, 'gender'] = x_train['gender'].map({'female': -1, 'male': 1, 'other': 0})
print(x_train['course'].unique())


y_train = train_data['exam_score']

x_test = test_data[['age', 'gender', 'course', 'study_hours', 'class_attendance',
       'internet_access', 'sleep_hours', 'sleep_quality', 'study_method',
       'facility_rating', 'exam_difficulty']]

