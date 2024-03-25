import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from joblib import dump
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

df = pd.read_csv("interstellar_travel.csv")

le = LabelEncoder()

df.dropna(inplace = True)
df.drop_duplicates(inplace = True)
df.drop("Star System", axis = 1, inplace = True)
df.drop("Booking Date", axis = 1, inplace = True)
df.drop("Departure Date", axis = 1, inplace = True)
df.drop("Customer Satisfaction Score", axis = 1, inplace = True)

df.drop(df[df["Price (Galactic Credits)"] < 0].index, inplace = True)

df = df[1000 : 100000]

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

# print(np.min(Y))

scores = cross_val_score(ElasticNet(fit_intercept = True), X, Y, cv = 8)

pca = PCA(n_components = 13)
pca.fit(X)

X = pca.transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.15, random_state=42
)

model = GridSearchCV(
    ElasticNet(fit_intercept = True),
    {
        'alpha': list(map(lambda x: x, range(1, 10))),
        'l1_ratio': list(map(lambda x: x / 10, range(1, 10))),
    },
    n_jobs=-1,
)

model.fit(X_train, Y_train)
print(model.best_params_)

model = model.best_estimator_
dump(model, "model.pkl")

print(r2_score(Y, model.predict(X)))
print(mean_absolute_error(Y, model.predict(X)))

Ypred = model.predict(X)
plt.boxplot(Y)
# plt.boxplot(Ypred)
plt.show()
# wR = []
# wP = []
# Ymm = []
# Ypmm = []

# for i in range(len(Y)):
#     wR.append(Y[i])
#     wP.append(Ypred[i])
#     if len(wR) > 5:
#         Ymm.append(sum(wR) / 5)
#         Ypmm.append(sum(wP) / 5)
#         wR.pop(0)
#         wP.pop(0)

# plt.plot(Ymm)
# plt.plot(Ypmm)
# plt.show()