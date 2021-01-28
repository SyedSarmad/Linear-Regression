# Linear-Regression
#Pandas is a library to manipulate data, the data is stored in a data structure called
#the DataFrame. it allows you to store data in a table format with columns containing data
# #and rows that are called observations which are your labels for the column
import pandas as pd

#creates multidimensional arrays that are used for calculations
import numpy as np

#library used to plot data on a graph
import matplotlib.pyplot as plt

#sklearn is a machine learning library, this specific one is
#used for linear regression machine learning
from sklearn.linear_model import LinearRegression

#implements several loss, score, and utility functions to measure classification performance
from sklearn.metrics import r2_score

#used to create statistical models
import statsmodels.api as sm

#using panda to read in the data and convert it into a DataFrame data structure
data = pd.read_csv("/Users/sarmad/dataforLR.csv")

#displays first 4 rows of data
print(data.head())
#deletes the column 'unnamed' because it is redundant data
data = data.drop(['Unnamed: 0'], axis = 1)
#displays first 4 rows of data with the removal of the 'unnamed' column
print(data.head())

#asigning the size we want to display the plot as. Width = 16 inch and Height = 8 inch
plt.figure(figsize = (16, 8))
#using the 'TV' as x plane and 'Sales' as the y plane. Also assigning a color blue to represent the dots
#using the scatter() method in 'matplotlib.pyplot' library to create a scatter plot
plt.scatter(data['TV'], data['sales'], c = 'blue')
plt.xlabel("Money spent on TV ads ($)")
plt.ylabel("Sales ($)")
plt.show()

#setting variable x to be the 'TV' value columns
#to my understanding the reshape method is telling it that we want second dimension size to be 1, the -1 is
#saying that we want the correct size of the 1st dimension given the second dimension
X = data['TV'].values.reshape(-1,1)
y = data['sales'].values.reshape(-1,1)

#creating a linear regression model object
reg = LinearRegression()
#passing data into the linear regression model object
reg.fit(X, y)
print("The linear model is: Y = {:.5} + {:.5}X".format(reg.intercept_[0], reg.coef_[0][0]))


predictions = reg.predict(X)
plt.figure(figsize = (16, 8))
plt.scatter(data['TV'], data['sales'], c = 'black')
plt.plot( data['TV'], predictions, c ='blue', linewidth = 2)
plt.xlabel("Money spent on TV ads ($)")
plt.ylabel("Sales ($)")
plt.show()
