from main import *


housing_test = model.predict(testSet)
mse = mean_squared_error(housing_label, testSet)
rmse = np.sqrt(mse)
print(f'error: {rmse}')

score = cross_val_score(model, housing_test, housing_label, scoring='neg_mean_squared_error')
rmse_score = np.sqrt(-score)
print(print_score(rmse_score))