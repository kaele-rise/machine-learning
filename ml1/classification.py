import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# импорт credit_data
data = np.array(pd.read_csv('credit_data'))
train_approved = data[:, 4] # целевая переменная
train_approved = np.where(train_approved == 1, 1, -1)
raw_data = data[:, :4] # признаки

scaler = StandardScaler()
raw_data = scaler.fit_transform(raw_data)
X = np.c_[np.ones(raw_data.shape[0]), raw_data] # добавление признака для w0
# print(train_approval)

w = np.random.normal(0, 0.1, 5) # инициализация весов
learning_rate = 0.001 # шаг изменения весов
max_iter = 1000 # максимальное кол-во итераций
data_size = X.shape[0] # размер выборки

predict = lambda x: np.sign(np.dot(w, x)) # модель

errors = []
for n in range(max_iter):
    total_error = 0
    indices = np.random.permutation(data_size) # перемешивание данных

    for i in range(data_size):
        prediction = predict(X[i]) # предсказание
        error = train_approved[i] * prediction # проверка ошибки (если знаки различны - error принимает значение < 0)

        if error < 0:
            w += learning_rate * train_approved[i] * X[i] # корректировка весов
            total_error += 1

    errors.append(total_error)

    if total_error == 0: # условие раннего выхода
        print(f'Обучение завершено на эпохе {n}')
        break

    if n % 100 == 0: # отображение результата каждые 100 эпох
        print(f'Эпоха: {n}, Кол-во ошибок: {total_error}')


# визуализация
plt.plot(errors)
plt.title('Кол-во ошибок по эпохам')
plt.xlabel('Эпоха')
plt.ylabel('Кол-во ошибок')
plt.show()

# оценка точности
correct = 0
for i in range(data_size):
    if predict(X[i]) == train_approved[i]:
        correct += 1

accuracy = correct / data_size
print(f"Точность на обучающей выборке: {accuracy:.2%}")
print(f"Веса модели: {w}")


