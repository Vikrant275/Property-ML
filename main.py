import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import *
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

def print_score(score):
    print("score: ", score)
    print("mean: ", score.mean())
    print("standard deviation: ", score.std())


housing = pd.read_csv("property.csv")
#print(housing.head())
#print(housing.info())
#print(housing['CHAS'].value_counts())
#print(housing.describe())
#print(housing.hist(bins=50, figsize=(20,18)))
#plt.show()
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    trainSet = housing.loc[train_index]
    testSet = housing.loc[test_index]

print(trainSet['CHAS'].value_counts())
print(testSet['CHAS'].value_counts())

housing = trainSet
print(housing.describe())
attribute = ['MEDV', 'RM', 'ZN', 'LSTAT']
#scatter_matrix(housing[attribute], alpha=1, figsize=(12,8))
#housing.plot(kind="scatter",x="RM", y="MEDV")
housing["TAXRM"]= housing["TAX"]/housing["RM"]
print(housing.head())
corr_matrix = housing.corr()
print(corr_matrix['MEDV'].sort_values(ascending = True))
housing.plot(kind='scatter', x='TAXRM', y='MEDV')
#plt.show()

housing_label = housing['MEDV'].copy()
housing = housing.drop('MEDV', axis = 1)
print(housing.describe())


my_pipeline = Pipeline([('imputer', SimpleImputer(strategy = 'median')), ('std_scaler', StandardScaler()),])
housing_tr = my_pipeline.fit_transform(housing)
#print(housing_tr.shape)

#model = LinearRegression()
#model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_tr, housing_label)
some_data = housing.iloc[:5]
some_label = housing_label[:5]
prepared_data = my_pipeline.transform(some_data)
print(model.predict(prepared_data))
print(some_label)


housing_pre = model.predict(housing_tr)
mse = mean_squared_error(housing_label, housing_pre)
rmse = np.sqrt(mse)
print(f'error: {rmse}')

score = cross_val_score(model, housing_tr, housing_label, scoring='neg_mean_squared_error')
rmse_score = np.sqrt(-score)
print(print_score(rmse_score))
