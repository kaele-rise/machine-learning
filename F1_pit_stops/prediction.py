from sklearn.model_selection import GroupKFold
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd


# обучение модели
def model_training(X, y, lgbm_params, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)

    models = []
    val_scores = []

    for fold, (train_index, val_index) in enumerate(gkf.split(X, y, groups=X['Race'])):
        print(f'Fold {fold+1}')

        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model = LGBMClassifier(**lgbm_params)

        model.fit(X_train, y_train)

        models.append(model)

        y_pred = model.predict_proba(X_val)[:, 1]
        roc_auc = roc_auc_score(y_val, y_pred)
        print(f'ROC AUC: {roc_auc}')
        val_scores.append(roc_auc)


    return models, val_scores


# предсказание вероятностей
def model_prediction(models, X_test):
    probas = np.zeros((X_test.shape[0], len(models)))
    for i, model in enumerate(models):
        probas[:, i] = model.predict_proba(X_test)[:, 1]

    return probas.mean(axis=1)


if __name__ == '__main__':
    train = pd.read_csv('train_aug.csv')
    test = pd.read_csv('test_aug.csv')

    print(len(train['Race'].unique()))

    X_train = train.drop(['id', 'PitNextLap'], axis=1)
    y_train = train['PitNextLap']

    cat_cols = ['Driver', 'Compound', 'Race']
    X_train[cat_cols] = X_train[cat_cols].astype('category')

    lgbm_params = {
        'n_estimators': 500,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': (len(y_train) - y_train.sum()) / y_train.sum(),  # балансировка классов
        'random_state': 42,
        'verbose': -1
    }

    models, val_scores = model_training(X_train, y_train, lgbm_params)


    X_test = test
    X_test = X_test.drop(['id'], axis=1)
    X_test[cat_cols] = X_test[cat_cols].astype('category')

    y_pred = model_prediction(models, X_test)

    submission = pd.DataFrame({
        'id': test['id'],
        'PitNextLap': y_pred
    })

    submission.to_csv('submission.csv', index=False)




