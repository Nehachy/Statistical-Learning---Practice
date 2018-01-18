import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AR
from pandas import Series
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
import pandas


file2 = '/Users/nehachoudhary/Desktop/test_time_series.csv'
data_df=pandas.read_csv(file2,index_col = False)
data_df['Date'] = pandas.to_datetime(data_df['Date'])
print(data_df)
model = AR(data_df.as_matrix)


'''
def read_data_power_plant():
        print('Reading power plant dataset ...')
        train_x = np.genfromtxt('/Users/nehachoudhary/Desktop/test_time_series.tst', skipheader)
        return train_x

def average(series):
    return float(sum(series))/len(series)

def moving_average(series, n):
    return average(series[-n:])


X = [1359,3171,1603,1275,1606,1552,1333,1449,1628,1759,2405,2589,3306,6014,7953]
train, test = X[1:len(X)-5], X[len(X)-5:]

#Print Basic average forecast
print(" Prediction =" , average(X))
# generate the next 2 forcast using mv
print(" Prediction 1 =" , moving_average(X, 3))
X1 = X + [moving_average(X, 3)]
print(" Prediction 2 =" , moving_average(X1, 3))
pred_mv = []

while True:
   p = moving_average(X, 10)
   X = X + [p]
   pred_mv.append(moving_average(X, 10))
   if len(pred_mv) == 5:
        break

print(pred_mv)
model = AR(train)
model_fit = model.fit( maxlag=7)
print('Lag: %s' % model_fit.k_ar)
print('Coefficients: %s' % model_fit.params)
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
for i in range(len(predictions)):
	print('predicted=%f, expected=%f' % (pred_mv[i], test[i]))

error = mean_squared_error(test, pred_mv)
print('Test MSE: %.3f' % error)
# plot results
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()


#m.predict(p, start=10, end=20, dynamic=False)
#series = Series.from_csv('/Users/nehachoudhary/Desktop/test_time_series.csv', header=0)
#print(series.head())
#series.plot()
#pyplot.show()

'''
