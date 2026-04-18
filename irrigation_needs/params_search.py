from sklearn.model_selection import StratifiedKFold
from prediction import data_preprocessor
from optuna.integration import LightGBMPruningCallback
import optuna
import lightgbm as lgb
from sklearn.metrics import f1_score
import numpy as np

X, y, test_data, feature_columns, cat_columns, num_columns = data_preprocessor()

def hyperparameter_search(X, y, cat_columns, n_trials=50, n_splits=5):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
            'objective': 'multiclass',
            'num_class': 3,
            'random_state': 42,
            'verbosity': -1,
            'n_jobs': -1
        }

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_scores = []

        for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            model = lgb.LGBMClassifier(**params)

            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='multi_logloss',
                callbacks=[
                    lgb.early_stopping(50),
                    lgb.log_evaluation(0),
                ],
                categorical_feature=cat_columns
            )

            y_pred = model.predict(X_val)
            score = f1_score(y_pred, y_val, average='macro')
            fold_scores.append(score)

            best_loss = model.best_score_['valid_0']['multi_logloss']
            trial.report(best_loss, step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(fold_scores)

    study = optuna.create_study(direction='maximize', study_name='lgbm_optimization')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_score = study.best_value

    print(f"\nЛучшие гиперпараметры: {best_params}")
    print(f"Лучшее значение f1 score: {best_score:.4f}")

    return best_params

best_params = hyperparameter_search(X, y, cat_columns, n_trials=50, n_splits=5)

print(best_params)


