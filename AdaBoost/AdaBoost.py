import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# import data
train_data = pd.read_csv('train.csv')
y_train = train_data['rainfall']
x_train = train_data[['day', 'pressure', 'maxtemp', 'temparature', 'mintemp',
       'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection',
       'windspeed']]

x_test = pd.read_csv('test.csv')

x_train, y_train, x_test = np.array(x_train), np.array(y_train), np.array(x_test)

y_train[y_train == 0] = -1
print(y_train[:15])


