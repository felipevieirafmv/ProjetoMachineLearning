import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from joblib import dump
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import seaborn as sns

def remove_outlier(column):
    standardDeviationDistance = np.std(df[column]) * 3
    meanDistance = np.mean(df[column])
    df.drop(df[np.abs(df[column] - meanDistance) > standardDeviationDistance].index, inplace = True)

df = pd.read_csv("interstellar_travel.csv")

le = LabelEncoder()

df.dropna(inplace = True)
df.drop_duplicates(inplace = True)
df.drop("Star System", axis = 1, inplace = True)
df.drop("Booking Date", axis = 1, inplace = True)
df.drop("Departure Date", axis = 1, inplace = True)
df.drop("Customer Satisfaction Score", axis = 1, inplace = True)

df.drop(df[df["Price (Galactic Credits)"] < 0].index, inplace = True)

remove_outlier("Distance to Destination (Light-Years)")
remove_outlier("Price (Galactic Credits)")

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
# print(X)

sns.heatmap(df.corr(), annot=True, fmt=".2f", vmin=-1, vmax=1,
            cmap=sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True))

scores = cross_val_score(GradientBoostingRegressor(), X, Y, cv = 8)
print(scores)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.5, random_state=42
)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

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

YpredTrain = est.predict(X_train)
print("Treino")
print(r2_score(Y_train, YpredTrain))
print(mean_absolute_error(Y_train, YpredTrain) / Y_train.mean())
print(mean_squared_error(Y_train, YpredTrain, squared = False) / Y_train.mean())

YpredTest = est.predict(X_test)
print("Teste")
print(r2_score(Y_test, YpredTest))
print(mean_absolute_error(Y_test, YpredTest) / Y_test.mean())
print(mean_squared_error(Y_test, YpredTest, squared = False) / Y_test.mean())

# dfTest = pd.read_csv("test.csv")

# print("Teste")
# print(est.predict(dfTest))
# print(r2_score([105, 102], est.predict(dfTest)))
# print(mean_absolute_error([105, 102], est.predict(dfTest)))
# print(mean_squared_error([105, 102], est.predict(dfTest)))

plt.scatter(Y_test, YpredTest)
plt.show()

# tuples = []

# Y_pred2 = list(est.predict(X_test))

# Y_test2 = list(Y_test)

# errorIndexes = []

# for aaa in range(len(Y_test2)):
#     if(abs(Y_pred2[aaa] - Y_test2[aaa]) > 100):
#         errorIndexes.append(aaa)

# for index in sorted(errorIndexes, reverse=True):
#     del Y_pred2[index]
#     del Y_test2[index]

# for bbb in range(len(Y_test2)):
#     tuples.append((Y_test2[bbb], Y_pred2[bbb]))

# ordened_tuples = sorted(tuples, key = lambda x: x[0])

# realY = []
# predY = []

# for value in ordened_tuples:
#     realY.append(value[0])
#     predY.append(value[1])

# print("Teste2")
# print(r2_score(Y_test2, Y_pred2))
# print(mean_absolute_error(Y_test2, Y_pred2))
# print(mean_squared_error(Y_test2, Y_pred2))

# plt.plot(realY)
# plt.plot(predY)
# plt.show()
