- 👋 Hi, I’m @lgiman
- 👀 I’m interested in ...
- 🌱 I’m currently learning ...
- 💞️ I’m looking to collaborate on ...
- 📫 How to reach me ...

<!---

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

#import file
df = pd.read_excel (r'D:\Other\Dars\Thesis\Down Grade to MSC\Data\Small\bOTH\small clear sample\small_clear_cleaned.xlsx')
df.head(6)
#OR
df = pd.read_csv("FuelConsumption.csv")
df.head(6)


#Subset a new data file
cdf = df[['treeid','site','bloc','family','ACA','category','RPC','moe','mor','den']]
cdf.head(9)
cdf.describe()

###### choose part of data as train and test
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk] # ~ not msk

################ Simple Regression Model
plt.scatter(train.Dens, train.MOE,  color='blue')
plt.xlabel("D")
plt.ylabel("MOE")
plt.show()

#####Using sklearn package to model data and obtain Tetha

from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['Dens']])
train_y = np.asanyarray(train[['MOE']])
regr.fit (train_x, train_y)
# The coefficients
print ('Coefficients: ', regr.coef_)   #tetha 1
print ('Intercept: ',regr.intercept_)  #tetha 0

# plot plus regression line
plt.scatter(train.Dens, train.MOE,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("D")
plt.ylabel("MOE")

#### test data
from sklearn.metrics import r2_score
test_x = np.asanyarray(test[['Dens']])
test_y = np.asanyarray(test[['MOE']])
test_y_ = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , test_y_) )



###### Multi regression
from sklearn import linear_model
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (x, y)
# The coefficients
print ('Coefficients: ', regr.coef_) 
print ('Intercept: ',regr.intercept_) 

y_hat = regr.predict(test[['ENGINESIZE', 'CYLINDERS','FUELCONSUMPTION_COMB']])
x = np.asanyarray(test[['ENGINESIZE', 'CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(test[['CO2EMISSIONS']])
print("Residual sum of squares: %.2f"
      % np.mean((y_hat - y)**2))
print('Variance dcore: %.2f' % regr.score(x,y))
# Variance score : 1 is perfect prediction



