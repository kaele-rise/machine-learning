import numpy as np
import pandas as pd

# 1к примеров с seed для воспроизводимости
np.random.seed(42)
n_samples = 1000

# генерация признаков (возраст, годовой доход (тыс. долларов), кредитный рейтинг, сумма кредита)
age = np.random.randint(18, 75, size=n_samples)
income = np.random.normal(70, 30, n_samples).clip(20, 200)
credit_score = np.random.normal(600, 150, n_samples).clip(300, 850)
loan_amount = np.random.uniform(5, 100, n_samples)

# условия одобрения кредита
approval = (
    (credit_score > 650) &
    (income > 50) &
    ((loan_amount / income) < 0.5)
).astype(int)

# сборка данных в .csv файл
data = pd.DataFrame({
    'Age': age,
    'Income': income,
    'Credit_score': credit_score,
    'Loan_amount': loan_amount,
    'Approved': approval
})

data['Income'] = data['Income'] + np.random.normal(0, 5, n_samples)
data.to_csv('credit_data', index=False)
