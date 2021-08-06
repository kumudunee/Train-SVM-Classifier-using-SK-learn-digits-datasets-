import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from sklearn import datasets

digits = load_digits()
print(dir(digits))

print(digits.data)

print(digits.target)

df = pd.DataFrame(digits.data)
print(df)

df['target'] = digits.target
print(df)

model = svm.SVC(kernel='linear',gamma=0.01,C=100)

print(len(digits.data))

x,y = digits.data[:-10],digits.target[:-10]

model.fit(x,y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print(len(x_train))
print(len(x_test))

print("Prediction of last:",model.predict(digits.data[[-5]]))

plt.imshow(digits.images[-5],cmap=plt.cm.gray_r,interpolation="nearest")

plt.show()

model.fit(x_train, y_train)
print(model)
print(model.score(x_test,y_test))
