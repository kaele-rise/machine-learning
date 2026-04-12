import pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, make_scorer
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)


# обработка данных
def data_preprocessor():
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')

    # аугментация данных
    train_data['Heat_Stress'] = ((train_data['Soil_Moisture'] < 22) &
                                 (train_data['Temperature_C'] > 30)) # стрессовые показатели влажности и темп-ры
    test_data['Heat_Stress'] = ((test_data['Soil_Moisture'] < 22) &
                                 (test_data['Temperature_C'] > 30))

    train_data['Irrigation_Efficiency'] = (train_data['Previous_Irrigation_mm'] /
                                           train_data['Field_Area_hectare']) # полив на гектар
    test_data['Irrigation_Efficiency'] = (test_data['Previous_Irrigation_mm'] /
                                           test_data['Field_Area_hectare'])

    train_data['Temp_Humidity_index'] = train_data['Humidity'] * train_data['Temperature_C'] # влажность * темп-ра
    test_data['Temp_Humidity_index'] = test_data['Humidity'] * test_data['Temperature_C']

    train_data['Solar_Wind_Stress'] = train_data['Wind_Speed_kmh'] * train_data['Sunlight_Hours'] # скорость ветра * солнце
    test_data['Solar_Wind_Stress'] = test_data['Wind_Speed_kmh'] * test_data['Sunlight_Hours']




    print(train_data.describe())
    print(train_data.columns)
    # print(train_data['Crop_Type'].unique())
    # print(train_data['Crop_Growth_Stage'].unique())

    # разметка столбцов (кат. / числ.)
    cat_columns = ['Soil_Type', 'Crop_Type', 'Crop_Growth_Stage',
                   'Season', 'Irrigation_Type', 'Water_Source',
                   'Mulching_Used', 'Region', 'Heat_Stress']
    train_data[cat_columns] = train_data[cat_columns].astype('category')
    test_data[cat_columns] = test_data[cat_columns].astype('category')

    num_columns = [col for col in train_data if (col not in cat_columns) and
                   (col != 'id') and
                   (col != 'Irrigation_Need')]

    # гистограмма каждого столбца
    # for col in num_columns:
    #     plt.hist(train_data[col], bins=30, edgecolor='black')
    #     plt.title(col)
    #     plt.show()


    # обучающие данные
    feature_columns = num_columns + cat_columns
    X = train_data[feature_columns]
    y = train_data['Irrigation_Need'].map({'Low': 0, 'Medium': 1, 'High': 2})

    return X, y, test_data, feature_columns, cat_columns, num_columns

# обучение модели
def model_training(X, y, cat_columns):
    # обучение модели (кросс-валидация)
    n_splits = 5 # кол-во разбиений кросс-вал-ции
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    models = []
    val_scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f'fold {fold+1}')

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(
            n_estimators=5000,
            learning_rate=0.05,
            objective='multiclass',
            num_class=3,
            random_state=42,
            verbosity=-1
        )

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
            eval_metric='multi_logloss',
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
            categorical_feature=cat_columns)

        y_pred = model.predict(X_val)
        # метрики на фолде
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='macro')

        print(f'accuracy: {acc}')
        print(f'f1: {f1}\n')


        models.append(model)
        val_scores.append(acc)

    return models

# предсказание
def model_prediction(test_data, feature_columns, models):
    # предсказание (ср. значение вероятности ансамбля моделей)
    X_test = test_data[feature_columns]

    test_probs = np.zeros((len(X_test), 3))
    for model in models:
        y_pred = model.predict_proba(X_test)
        test_probs += y_pred / len(models)


    predicted_class_idx = np.argmax(test_probs, axis=1)
    class_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
    predicted_labels = [class_mapping[idx] for idx in predicted_class_idx]

    submission = pd.DataFrame({
        'id': test_data['id'],
        'Irrigation_Need': predicted_labels
    })
    submission.to_csv('submission.csv', index=False)


if __name__ == "__main__":
    X, y, test_data, feature_columns, cat_columns, num_columns = data_preprocessor()
    models = model_training(X, y, cat_columns)
    model_prediction(test_data, feature_columns, models)