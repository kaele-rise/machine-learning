import pandas as pd
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

cat_columns = ['Soil_Type', 'Crop_Type', 'Crop_Growth_Stage',
               'Season', 'Irrigation_Type', 'Water_Source',
               'Mulching_Used', 'Region']
train_data[cat_columns] = train_data[cat_columns].astype('category')
test_data[cat_columns] = test_data[cat_columns].astype('category')
num_columns = [col for col in train_data if (col not in cat_columns) and
               (col != 'id') and
               (col != 'Irrigation_Need')]

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

feature_columns = num_columns + cat_columns
X = train_data[feature_columns]
y = train_data['Irrigation_Need'].map({'Low': 0, 'Medium': 1, 'High': 2})


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
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='macro')

    print(f'accuracy: {acc}')
    print(f'f1: {f1}\n')


    models.append(model)
    val_scores.append(acc)



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
