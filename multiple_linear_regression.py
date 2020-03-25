#import the necessary packages 
import matplotlib.pyplot as plt 
import pandas as pd 
import pylab as pl
import numpy as np 

df = pd.read_csv('FuelConsumptionCo2.csv')

#data exploration
print(df.describe())
print(df.columns)
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
print(cdf.head(9))

viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()

# plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
# plt.xlabel("FUELCONSUMPTION_COMB")
# plt.ylabel("Emission")
# plt.show()

# plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
# plt.xlabel("Engine size")
# plt.ylabel("Emission")
# plt.show()

# plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS,  color='blue')
# plt.xlabel("Cylinder")
# plt.ylabel("Emission")
# plt.show()
#Lets split our dataset into train and test sets, 80% of the entire data for training, and the 20% 
#for testing. We create a mask to select random rows using np.random.rand() function: 
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

from sklearn import linear_model
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (x, y)
# The coefficients
print ('Coefficients: ', regr.coef_)


