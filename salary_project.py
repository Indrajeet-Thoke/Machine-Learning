import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

#Load the dataset
dataset = pd.read_csv(r"C:\Users\DELL\Desktop\Practice On ML\Salary_Data.csv")

#Split the data into independent and dependent variables
X = dataset.iloc[:,:-1]
Y = dataset.iloc[:, -1]

#split the dataset into training and testing sets(80-20%)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=0)

#Train the model
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#predict the test set
Y_pred = regressor.predict(X_test)

#comparision for Y_test vs Y_pred
comparison = pd.DataFrame({"Actual" : Y_test, "Prediced": Y_pred})
print(comparison)

#visualization the Tranning
plt.scatter(X_test, Y_test, color = "red")
plt.plot(X_train, regressor.predict(X_train), color = "blue")
plt.title("Salary vs Experience (Test set)")
plt.xlabel("Years of Expeience")
plt.ylabel("Salary")
plt.show()
#Predict Salary for 12 and 20 years of experience using the trained model
Y_12 = regressor.predict([[12]])
Y_20 = regressor.predict([[20]])
print(f"Predicted salary for 12 years of experience: ${Y_12[0]:,.2f}")
print(f"Predicted salary for 20 years of experience: ${Y_20[0]:,.2f}")

#Check the model Performance

bias = regressor.score(X_train, Y_train)
variance = regressor.score(X_test, Y_test)
train_mse = mean_squared_error(Y_train, regressor.predict(X_train))

print(f"Training score (R^2): {bias:.2f}")
print(f"Testing score (R^2): {variance:.2f}")
print(f"TRaining MSE: {train_mse:.2f}")

#save the tain model to desk

import pickle
filename = "linear_regression_model.pkl"
with open(filename, "wb") as file:
    pickle.dump(regressor, file)
print("Model has been pickled and saved as linear_regression_model.pkl")

import os
print(os.getcwd())

