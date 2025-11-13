import numpy as np
import pandas as pd

# 1к примеров с seed для воспроизводимости
np.random.seed(42)
n_samples = 1000

# генерация признаков (возраст, годовой доход (тыс. долларов), кредитный рейтинг, сумма кредита)
# числовые признаки
age = np.random.randint(18, 75, size=n_samples)
income = np.random.normal(70, 30, n_samples).clip(20, 200)
credit_score = np.random.normal(600, 150, n_samples).clip(300, 850)
loan_amount = np.random.uniform(5, 100, n_samples)

# категориальные признаки
education = np.random.choice(['high_school', 'bachelor', 'master', 'phd'], n_samples, p=[0.3, 0.4, 0.2, 0.1])
employment_status = np.random.choice(['unemployed', 'employed', 'self_employed', 'retired'], n_samples, p=[0.1, 0.6, 0.2, 0.1])
home_ownership = np.random.choice(['rent', 'mortgage', 'own', 'other'], n_samples, p=[0.3, 0.4, 0.25, 0.05])

# условия одобрения кредита
approval_probability = (
        (credit_score > 650) * 0.3 +
        (income > 50) * 0.2 +
        (home_ownership == 'own') * 0.1 +
        (employment_status == 'employed') * 0.2
)

approved = (approval_probability + np.random.normal(0, 0.1, n_samples)) > 0.5

# сборка данных в .csv файл
data = pd.DataFrame({
    'Age': age,
    'Income': income,
    'Credit_score': credit_score,
    'Loan_amount': loan_amount,
    'Education': education,
    'Employment_status': employment_status,
    'Home_ownership': home_ownership,
    'Approved': approved
})

data['Income'] = data['Income'] + np.random.normal(0, 5, n_samples)
data.to_csv('credit_data', index=False)
