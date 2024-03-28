import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from joblib import dump
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor

df = pd.read_csv("interstellar_travel.csv")

le = LabelEncoder()

df.dropna(inplace = True)
df.drop_duplicates(inplace = True)
df.drop("Star System", axis = 1, inplace = True)
df.drop("Booking Date", axis = 1, inplace = True)
df.drop("Departure Date", axis = 1, inplace = True)
df.drop("Customer Satisfaction Score", axis = 1, inplace = True)

df.drop(df[df["Price (Galactic Credits)"] < 0].index, inplace = True)

df = df[1000 : 2000]

df["Gender"] = le.fit_transform(df["Gender"])
df["Occupation"] = le.fit_transform(df["Occupation"])
df["Travel Class"] = le.fit_transform(df["Travel Class"])
df["Destination"] = le.fit_transform(df["Destination"])
df["Purpose of Travel"] = le.fit_transform(df["Purpose of Travel"])
df["Transportation Type"] = le.fit_transform(df["Transportation Type"])
df["Special Requests"] = le.fit_transform(df["Special Requests"])
df["Loyalty Program Member"] = le.fit_transform(df["Loyalty Program Member"])

Y = df["Price (Galactic Credits)"]
X = df.drop("Price (Galactic Credits)", axis = 1)

print(X)
