'''
given the data which consists the initial position,initial velocity,final position and the time taken to come to the final positiom,we can train a 
model which gives the coefficients which relate to the final position, from this we can interpret many things,we get the value of accelaration
due to gravity,we find that its motion dosen't really depend on it's mass as the coefficient corresponding to mass is 0,and we can get the equation
of motion from it,it's just like rediscovering the equations of motion.
'''
import numpy as np
import pandas as pd
import csv 
import math, random


def location(x_0,v_0,t):
    x=x_0+v_0*t-(9.8/2)*t**2
    
    return x


with open('location.csv',mode='w') as gravity_file:
    gravity_writer=csv.writer(gravity_file,delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
    gravity_writer.writerow(['initial_position', 'initial_velocity', 'mass', 'time', 'location'])
    for i in range(1000):
        initial_position=random.randrange(1,10000)
        initial_velocity=random.randrange(1,100)
        mass=random.randrange(1,1000)
        time=random.randrange(1,100)
        gravity_writer.writerow([initial_position, initial_velocity, mass, time, location(initial_position, initial_velocity, time)])


#Dataset
dataset=pd.read_csv("location.csv")

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

regressor=LinearRegression()
#we are gonna consider polynomials upto 2 degree.
poly=PolynomialFeatures(degree=3)
x=poly.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

regressor.fit(x_train,y_train)


coeff=regressor.coef_
b=regressor.intercept_

coeff=np.around(coeff,decimals=2)

print(coeff)

