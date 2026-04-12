from prediction import data_preprocessor
from sklearn.model_selection import RandomizedSearchCV,StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, make_scorer
import lightgbm as lgb

X, y, test_data, feature_columns, cat_columns, num_columns = data_preprocessor()

X_sample, _, y_sample, _ = train_test_split(
    X, y, train_size=0.15, stratify=y, random_state=42
)

# подбор гиперпараметров
base_params = {
    'objective': 'multiclass',
    'num_class': 3,
    'random_state': 42,
    'verbosity': -1,
    'n_jobs': -1,
    'n_estimators': 300
}

param_dist = {
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'num_leaves': [20, 31, 40, 50],
    'max_depth': [-1, 5, 10, 15],
    'min_child_samples': [10, 20, 30, 50],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'reg_alpha': [0, 0.01, 0.1, 1],
    'reg_lambda': [0, 0.01, 0.1, 1]
}

lgb_model = lgb.LGBMClassifier(**base_params)
cv_inner = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
scorer = make_scorer(f1_score, average='macro')


random_search = RandomizedSearchCV(
    estimator=lgb_model,
    param_distributions=param_dist,
    n_iter=10,
    scoring=scorer,
    cv=cv_inner,
    random_state=42,
    verbose=1,
    n_jobs=-1
)

random_search.fit(X_sample, y_sample, categorical_feature=cat_columns)

print(f"Лучшие параметры: {random_search.best_params_}")
print(f"Лучший F1-macro: {random_search.best_score_:.4f}")

model_params = random_search.best_params_