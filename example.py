from joblib import load
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("romantismo.csv")
le = LabelEncoder()

df.drop("CompatibilityScore", axis=1, inplace=True)
df.dropna(inplace=True)

# df["AgeCategory"] = le.fit_transform(df["AgeCategory"])

Y = df["Response"]
X = df.drop("Response", axis=1)

loaded_model = load(open("best.pkl", 'rb'))

print(X)

for i in range(100): 
    predict = loaded_model.predict(X)
    print("eliana: ", predict[0])
    print("ana: ", predict[1])
    print("emyli: ", predict[2])
    print("mateus: ", predict[3])
    print("trevis: ", predict[4])
    print("lander: ", predict[5])
    print("juan: ", predict[6])

plt.scatter(x=np.arange(Y.size), y=Y, s=20)
plt.scatter(x=np.arange(predict.size), y=predict, s=10)
plt.show()