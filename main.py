import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from joblib import dump
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet
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

scores = cross_val_score(GradientBoostingRegressor(), X, Y, cv = 8)
print(scores)

pca = PCA(n_components = 13)
pca.fit(X)

X = pca.transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.15, random_state=42
)

est = GridSearchCV(
    GradientBoostingRegressor(),
        {
            "n_estimators": [10000],
            "learning_rate": [0.1],
            "max_depth": [2],
            "random_state": [42],
            "loss": ["absolute_error"]
        },
        n_jobs = -1
)

est.fit(X_train, Y_train)
print(est.best_params_)

dump(est, "est.pkl")

print("Treino")
print(r2_score(Y_train, est.predict(X_train)))
print(mean_absolute_error(Y_train, est.predict(X_train)))
print(mean_squared_error(Y_train, est.predict(X_train)))

print("Teste")
print(r2_score(Y_test, est.predict(X_test)))
print(mean_absolute_error(Y_test, est.predict(X_test)))
print(mean_squared_error(Y_test, est.predict(X_test)))

tuples = []

Y_pred2 = list(est.predict(X_test))

Y_test2 = list(Y_test)

for aaa in range(len(Y_test2)):
    if(Y_pred2[aaa] * Y_pred2[aaa] - Y_test2[aaa] * Y_test2[aaa] < 1000):
        Y_pred2.remove(Y_pred2[aaa])
        Y_test2.remove(Y_test2[aaa])
        aaa = aaa - 1

for bbb in range(len(Y_test2)):
    tuples.append((Y_test2[bbb], Y_pred2[bbb]))

ordened_tuples = sorted(tuples, key = lambda x: x[0])

realY = []
predY = []

for value in ordened_tuples:
    realY.append(value[0])
    predY.append(value[1])

print("Teste2")
print(r2_score(Y_test2, Y_pred2))
print(mean_absolute_error(Y_test2, Y_pred2))
print(mean_squared_error(Y_test2, Y_pred2))

plt.plot(realY)
plt.plot(predY)
plt.show()

