import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from joblib import dump
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.neighbors import KNeighborsRegressor

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

scores = cross_val_score(GradientBoostingRegressor(), X, Y, cv = 8)
print(scores)

pca = PCA(n_components = 13)
pca.fit(X)

X = pca.transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.15, random_state=42
)

est = GradientBoostingRegressor(
    n_estimators=10000,
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
    loss='squared_error'
)

estimators = [('ridge', RidgeCV()),
              ('lasso', LassoCV(random_state=42)),
              ('knr', KNeighborsRegressor(n_neighbors=20,
                                          metric='euclidean'))]

reg = StackingRegressor(
    estimators=estimators,
    final_estimator=est)

reg.fit(X_train, Y_train)

dump(est, "est.pkl")

print("Treino")
print(r2_score(Y_train, reg.predict(X_train)))
print(mean_absolute_error(Y_train, reg.predict(X_train)))

print("Teste")
print(r2_score(Y_test, reg.predict(X_test)))
print(mean_absolute_error(Y_test, reg.predict(X_test)))
