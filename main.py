import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split
from math import sqrt

dataset = pd.read_csv('weight_dataset.csv')
dataset['Height']=dataset.Height.apply(lambda val : 2.54*val)
dataset['Weight']=dataset.Weight.apply(lambda val : 0.45359237*val)

print(dataset.head(10))
print(dataset.info())
print(dataset.describe())

sns.scatterplot(x='Height',y='Weight',hue='Gender',data=dataset)

dataset['Gender'].replace('Female',0, inplace=True)
dataset['Gender'].replace('Male',1, inplace=True)

x_train,x_test, y_train,y_test = train_test_split(dataset.drop('Weight',axis=1), dataset.Weight, test_size=0.2, random_state=101)
regressor = XGBRegressor(n_estimators=100)
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)
print(f'MAE = {mean_absolute_error(y_test,y_pred)}')
print(f'XG Boost Regressor is about {round(regressor.score(x_test,y_test)*100)}% accurate!')


plt.show()