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


################## pOLYNOMIAL

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])


poly = PolynomialFeatures(degree=2) #daraje 2, y= x^0 +x^1 + x^2
train_x_poly = poly.fit_transform(train_x)
train_x_poly

clf = linear_model.LinearRegression()
train_y_ = clf.fit(train_x_poly, train_y)
# The coefficients
print ('Coefficients: ', clf.coef_)
print ('Intercept: ',clf.intercept_)



plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
XX = np.arange(0.0, 10.0, 0.1)
yy = clf.intercept_[0]+ clf.coef_[0][1]*XX+ clf.coef_[0][2]*np.power(XX, 2)
plt.plot(XX, yy, '-r' )
plt.xlabel("Engine size")
plt.ylabel("Emission")

#### Evaluation
from sklearn.metrics import r2_score

test_x_poly = poly.fit_transform(test_x)
test_y_ = clf.predict(test_x_poly)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y,test_y_ ) )


##### NonLinear model
#plot
plt.scatter(df.Year, df.Value, color='red')
#OR
plt.figure(figsize=(8,5))
x_data, y_data = (df["Year"].values, df["Value"].values)
plt.plot(x_data, y_data, 'ro')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()


#choosing a model. Some other types of non-linear functions are:

#1- Quadratic               Y= x^2
#2- Exponential             Y= bc^x
#3- Logarithmic             Y= log(x)     human height   
#4- Sigmoidal/Logistic      Y= a +(b /1 + e^(x - d)    S form


X = np.arange(-5.0, 5.0, 0.1)
Y = 1.0 / (1.0 + np.exp(-X))

plt.plot(X,Y) 
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()
