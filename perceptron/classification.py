import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# загрузка credit_data и разделение на Х и у
data = pd.read_csv('credit_data')

X = data.drop('Approved', axis=1)
y = data['Approved']
y = np.where(y == 1, 1, -1)


# модель классификации (перцептрон)
class Perceptron(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate = 0.01, max_iter = 1000, seed = 42):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.errors = []
        self.seed = seed

    def fit(self, X, y):
        np.random.seed(self.seed)

        #
        X_bias = np.c_[np.ones(X.shape[0]), X]

        # инициализация весов
        self.w = np.random.normal(0, 0.01, X_bias.shape[1])

        batch_size = 20

        n_samples = X_bias.shape[0]


        for n in range(self.max_iter):
            total_error = 0

            k = np.random.randint(0, n_samples - batch_size)
            x_batch = X_bias[k:(k + batch_size)]
            y_batch = y[k:(k + batch_size)]

            prediction = np.dot(x_batch, self.w)

            error = y_batch - prediction

            grad = np.dot(x_batch.T, error)

            self.w += self.learning_rate * grad

            total_error = np.sum(np.sign(prediction) != y_batch)

            if total_error == 0: # условие раннего выхода
                print(f'Обучение завершено на эпохе {n}')
                break

            if n % 100 == 0: # отображение результата каждые 100 эпох
                print(f'Эпоха: {n}, Кол-во ошибок: {total_error}')

        return self

    def predict(self, X):
        X_bias = np.c_[np.ones(X.shape[0]), X]
        predictions = np.dot(X_bias, self.w)
        return np.sign(predictions)


num_columns = ['Age', 'Income', 'Credit_score', 'Loan_amount']
categorical_columns = ['Education', 'Employment_status', 'Home_ownership']


# создание пайплайнов
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
]) #

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_columns),
    ('cat', categorical_pipeline, categorical_columns)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', Perceptron(learning_rate=0.01, max_iter=1000))
])

# разбиение выборки на обучающую и тестовую
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# обучение модели
pipeline.fit(X_train, y_train)

# визуализация ошибок по эпохам
plt.figure(figsize=(10, 6))
plt.plot(pipeline.named_steps['model'].errors)
plt.title('Количество ошибок по эпохам')
plt.xlabel('Эпоха')
plt.ylabel('Количество ошибок')
plt.grid(True)
plt.show()

y_pred = pipeline.predict(X_test)

# оценка качества
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary', pos_label=1)
recall = recall_score(y_test, y_pred, average='binary', pos_label=1)
f1 = f1_score(y_test, y_pred, average='binary', pos_label=1)

print("Оценка качества:")
print(f"Точность: {accuracy:.2%}")
print(f"Точность положительных прогнозов: {precision:.2%}")
print(f"Полнота: {recall:.2%}")
print(f"F1-Score: {f1:.2%}")

# отображение весов модели
print(f"\nВеса модели: {pipeline.named_steps['model'].w}")


